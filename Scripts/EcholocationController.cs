using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using UnityEngine.Rendering;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace StarterAssets
{
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct PointData
    {
        public Vector3 position;
        public Vector3 normal;
        public float startTime;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SimpleRaycastHit
    {
        public Vector3 p;
        public Vector3 normal;
        public float distance;
        public int colliderInstanceID;
    }



    public class EcholocationController : MonoBehaviour
    {
        // --- MODIFIED: The job tracker now needs its own data ---
        private struct PendingWave
        {
            public JobHandle jobHandle;
            public int waveBufferIndex;
            public NativeArray<RaycastHit> results; // Each job has its own results array
            public float triggerTime; // Store the time the wave was fired
        }

        [Header("Configuration")]
        public int rayCount = 50000;
        public float maxDistance = 100f;
        public float pointLifetime = 3.0f;
        [Tooltip("The maximum number of waves to stack before overwriting the oldest.")]
        public int maxWaves = 5;

        [Header("References")]
        public ComputeShader computeShader;
        public Mesh pointMesh;
        public Material pointMaterial;

        [Header("Wave Settings")]
        public float propagationSpeed = 50f;

        private Queue<PendingWave> pendingWaves;

        // Buffer Pool for GPU data
        private ComputeBuffer[] wavePointBuffers;
        private ComputeBuffer[] drawArgsBuffers;
        private uint[] drawArgsTemplate;

        // --- NEW: Pool of NativeArrays for CPU jobs ---
        private NativeArray<RaycastCommand>[] commandsPool;
        private NativeArray<RaycastHit>[] resultsPool;

        // Shared input buffer for the compute shader
        //private ComputeBuffer raycastHitsBuffer;
        private ComputeBuffer[] raycastHitsBufferPool;

        private int nextWaveIndex = 0;
        private Material instantiatedMaterial;
        private ComputeShader instantiatedComputeShader;

        private MaterialPropertyBlock propertyBlock; // Add this line

        private static readonly int processedPointsBufferID = Shader.PropertyToID("_PointsBuffer"),
            propagationSpeedID = Shader.PropertyToID("_PropagationSpeed"),
            timeID = Shader.PropertyToID("_Time"),
            rayCountID = Shader.PropertyToID("_RayCount"),
            lifetimeID = Shader.PropertyToID("_LifeTime"),
            raycastHitsBufferID = Shader.PropertyToID("_RaycastHitsBuffer"),
            processedPointsOutBufferID = Shader.PropertyToID("_ProcessedPointsBuffer");
        private int kernelIndex;

        void Start()
        {
            instantiatedMaterial = new Material(pointMaterial);
            propertyBlock = new MaterialPropertyBlock(); // Add this line
            instantiatedComputeShader = Instantiate(computeShader);
            pendingWaves = new Queue<PendingWave>();

            // --- Initialize GPU pools ---
            wavePointBuffers = new ComputeBuffer[maxWaves];
            drawArgsBuffers = new ComputeBuffer[maxWaves];
            drawArgsTemplate = new uint[5] { 0, 0, 0, 0, 0 };
            drawArgsTemplate[0] = pointMesh.GetIndexCount(0);
            drawArgsTemplate[2] = pointMesh.GetIndexStart(0);
            drawArgsTemplate[3] = pointMesh.GetBaseVertex(0);

            for (int i = 0; i < maxWaves; i++)
            {
                wavePointBuffers[i] = new ComputeBuffer(rayCount, sizeof(float) * 7, ComputeBufferType.Append);
                drawArgsBuffers[i] = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);
                drawArgsBuffers[i].SetData(drawArgsTemplate);
            }

            // --- Initialize CPU Job Data Pools ---
            commandsPool = new NativeArray<RaycastCommand>[maxWaves];
            resultsPool = new NativeArray<RaycastHit>[maxWaves];
            for (int i = 0; i < maxWaves; i++)
            {
                commandsPool[i] = new NativeArray<RaycastCommand>(rayCount, Allocator.Persistent);
                resultsPool[i] = new NativeArray<RaycastHit>(rayCount, Allocator.Persistent);
            }

            int simpleRaycastHitStride = sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) + sizeof(int);
            raycastHitsBufferPool = new ComputeBuffer[maxWaves];
            for (int i = 0; i < maxWaves; i++)
            {
                raycastHitsBufferPool[i] = new ComputeBuffer(rayCount, simpleRaycastHitStride, ComputeBufferType.Default);
            }

            kernelIndex = instantiatedComputeShader.FindKernel("GeneratePoints");
            //instantiatedComputeShader.SetBuffer(kernelIndex, raycastHitsBufferID, raycastHitsBuffer);
            instantiatedMaterial.SetFloat(lifetimeID, pointLifetime);
        }

        void OnDestroy()
        {
            // Complete any running job before destroying anything
            if (pendingWaves.Count > 0)
            {
                pendingWaves.Peek().jobHandle.Complete();
            }

            // Dispose all pooled NativeArrays
            for (int i = 0; i < maxWaves; i++)
            {
                if (commandsPool[i].IsCreated) commandsPool[i].Dispose();
                if (resultsPool[i].IsCreated) resultsPool[i].Dispose();
            }

            for (int i = 0; i < maxWaves; i++)
            {
                raycastHitsBufferPool[i]?.Release();
                wavePointBuffers[i]?.Release();
                drawArgsBuffers[i]?.Release();
            }

            if (instantiatedMaterial != null) Destroy(instantiatedMaterial);
            if (instantiatedComputeShader != null) Destroy(instantiatedComputeShader);
        }

        void Update()
        {
            // --- CONSUMER ---
            if (pendingWaves.Count > 0 && pendingWaves.Peek().jobHandle.IsCompleted)
            {
                var finishedWave = pendingWaves.Dequeue();
                finishedWave.jobHandle.Complete();

                // --- MODIFIED: Get the dedicated buffers for this specific wave ---
                ComputeBuffer currentPointBuffer = wavePointBuffers[finishedWave.waveBufferIndex];
                ComputeBuffer currentArgsBuffer = drawArgsBuffers[finishedWave.waveBufferIndex];
                ComputeBuffer currentHitsBuffer = raycastHitsBufferPool[finishedWave.waveBufferIndex]; // Get the right hits buffer
                NativeArray<RaycastHit> currentResults = finishedWave.results;

                // ... (your for-loop to create simpleHits remains the same) ...
                SimpleRaycastHit[] simpleHits = new SimpleRaycastHit[rayCount];
                for (int i = 0; i < rayCount; i++)
                {
                    var hit = currentResults[i];
                    simpleHits[i].p = hit.point;
                    simpleHits[i].normal = hit.normal;
                    simpleHits[i].distance = hit.distance;
                    simpleHits[i].colliderInstanceID = hit.collider != null ? hit.collider.GetInstanceID() : 0;
                }

                // --- MODIFIED: Use the dedicated buffer ---
                currentHitsBuffer.SetData(simpleHits); // Set data on the correct buffer

                currentPointBuffer.SetCounterValue(0);

                // --- MODIFIED: Tell the compute shader which buffers to use for this dispatch ---
                instantiatedComputeShader.SetBuffer(kernelIndex, raycastHitsBufferID, currentHitsBuffer); // Input
                instantiatedComputeShader.SetBuffer(kernelIndex, processedPointsOutBufferID, currentPointBuffer); // Output

                instantiatedComputeShader.SetFloat(propagationSpeedID, propagationSpeed);
                instantiatedComputeShader.SetFloat(timeID, finishedWave.triggerTime);
                instantiatedComputeShader.SetInt(rayCountID, rayCount);

                int threadGroups = Mathf.CeilToInt(rayCount / 64.0f);
                instantiatedComputeShader.Dispatch(kernelIndex, threadGroups, 1, 1);
                ComputeBuffer.CopyCount(currentPointBuffer, currentArgsBuffer, sizeof(uint));
            }

            // Trigger
            if (StarterAssetsInputs.Instance?.GetFireInputDown() ?? false)
            {
                TriggerEcholocation();
            }

            // Render
            for (int i = 0; i < maxWaves; i++)
            {
                // Get the buffer for the current wave
                var currentWaveBuffer = wavePointBuffers[i];

                // Set the buffer on the property block, NOT the material
                propertyBlock.SetBuffer(processedPointsBufferID, currentWaveBuffer);

                // Pass the property block to the draw call
                Graphics.DrawMeshInstancedIndirect(
                    pointMesh, 0, instantiatedMaterial,
                    new Bounds(Vector3.zero, new Vector3(1000.0f, 1000.0f, 1000.0f)),
                    drawArgsBuffers[i],
                    0, // argsOffset
                    propertyBlock // The property block with our override
                );
            }
        }

        // --- PRODUCER ---
        public void TriggerEcholocation()
        {
            // Get the specific command and results arrays for this new job from our pools
            var currentCommands = commandsPool[nextWaveIndex];
            var currentResults = resultsPool[nextWaveIndex];

            Vector3 origin = transform.position;
            for (int i = 0; i < rayCount; i++)
            {
                currentCommands[i] = new RaycastCommand(origin, UnityEngine.Random.onUnitSphere, QueryParameters.Default, maxDistance);
            }

            // Schedule the job with its own unique data, breaking the dependency chain
            var handle = RaycastCommand.ScheduleBatch(currentCommands, currentResults, 1);

            pendingWaves.Enqueue(new PendingWave
            {
                jobHandle = handle,
                waveBufferIndex = nextWaveIndex,
                results = currentResults, // Pass this specific results array along
                triggerTime = Time.time // Record the exact time of the click
            });

            nextWaveIndex = (nextWaveIndex + 1) % maxWaves;

            JobHandle.ScheduleBatchedJobs();
        }
    }
}