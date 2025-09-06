using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace StarterAssets
{
    // Serializable struct for Inspector color mapping (no changes here)
    [System.Serializable]
    public struct TagColor
    {
        public string tag;
        public Color color;
    }

    // The PointData struct now includes color (no changes from last version)
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct PointData
    {
        public Vector3 position;
        public float startTime;
        public Color color;
    }

    // The SimpleRaycastHit struct includes color (no changes from last version)
    [StructLayout(LayoutKind.Sequential)]
    public struct SimpleRaycastHit
    {
        public Vector3 p;
        public Vector3 normal;
        public float distance;
        public int colliderInstanceID;
        public Color color;
    }

    public class EcholocationController : MonoBehaviour
    {
        // --- (PendingWave struct is unchanged) ---
        private struct PendingWave
        {
            public JobHandle jobHandle;
            public int waveBufferIndex;
            public NativeArray<RaycastHit> results;
            public float triggerTime;
        }

        [Header("Configuration")]
        public int rayCount = 50000;
        public float maxDistance = 100f;
        public float pointLifetime = 3.0f;
        [Tooltip("The maximum number of waves to stack before overwriting the oldest.")]
        public int maxWaves = 5;

        [Header("References")]
        public ComputeShader computeShader;
        // MODIFIED: We need a mesh to instance. Assign a simple quad here.
        public Mesh pointMesh;
        // This material will be one created from your new Shader Graph
        public Material pointMaterial;

        [Header("Wave Settings")]
        public float propagationSpeed = 50f;

        [Header("Color Mapping")]
        public List<TagColor> tagColors;
        public Color defaultColor = Color.white;
        private Dictionary<string, Color> colorMap;

        // --- (Private variables are mostly the same) ---
        private Queue<PendingWave> pendingWaves;
        private ComputeBuffer[] wavePointBuffers;
        private ComputeBuffer[] drawArgsBuffers;
        private uint[] drawArgsTemplate;
        private NativeArray<RaycastCommand>[] commandsPool;
        private NativeArray<RaycastHit>[] resultsPool;
        private ComputeBuffer[] raycastHitsBufferPool;
        private int nextWaveIndex = 0;
        private Material instantiatedMaterial;
        private ComputeShader instantiatedComputeShader;
        private MaterialPropertyBlock propertyBlock;
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
            colorMap = new Dictionary<string, Color>();
            foreach (var tagColor in tagColors)
            {
                if (!colorMap.ContainsKey(tagColor.tag)) colorMap.Add(tagColor.tag, tagColor.color);
            }

            instantiatedMaterial = new Material(pointMaterial);
            propertyBlock = new MaterialPropertyBlock();
            instantiatedComputeShader = Instantiate(computeShader);
            pendingWaves = new Queue<PendingWave>();

            wavePointBuffers = new ComputeBuffer[maxWaves];
            drawArgsBuffers = new ComputeBuffer[maxWaves];

            // MODIFIED: The arguments for DrawMeshInstancedIndirect need 5 uints.
            drawArgsTemplate = new uint[5] { 0, 0, 0, 0, 0 };
            if (pointMesh != null)
            {
                drawArgsTemplate[0] = pointMesh.GetIndexCount(0);
                drawArgsTemplate[2] = pointMesh.GetIndexStart(0);
                drawArgsTemplate[3] = pointMesh.GetBaseVertex(0);
            }

            for (int i = 0; i < maxWaves; i++)
            {
                wavePointBuffers[i] = new ComputeBuffer(rayCount, Marshal.SizeOf<PointData>(), ComputeBufferType.Append);
                drawArgsBuffers[i] = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);
                drawArgsBuffers[i].SetData(drawArgsTemplate);
            }

            commandsPool = new NativeArray<RaycastCommand>[maxWaves];
            resultsPool = new NativeArray<RaycastHit>[maxWaves];
            for (int i = 0; i < maxWaves; i++)
            {
                commandsPool[i] = new NativeArray<RaycastCommand>(rayCount, Allocator.Persistent);
                resultsPool[i] = new NativeArray<RaycastHit>(rayCount, Allocator.Persistent);
            }

            int simpleRaycastHitStride = Marshal.SizeOf<SimpleRaycastHit>();
            raycastHitsBufferPool = new ComputeBuffer[maxWaves];
            for (int i = 0; i < maxWaves; i++)
            {
                raycastHitsBufferPool[i] = new ComputeBuffer(rayCount, simpleRaycastHitStride, ComputeBufferType.Default);
            }

            kernelIndex = instantiatedComputeShader.FindKernel("GeneratePoints");
            instantiatedMaterial.SetFloat(lifetimeID, pointLifetime);
        }

        void OnDestroy()
        {
            if (pendingWaves.Count > 0)
            {
                pendingWaves.Peek().jobHandle.Complete();
            }

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
            if (pendingWaves.Count > 0 && pendingWaves.Peek().jobHandle.IsCompleted)
            {
                var finishedWave = pendingWaves.Dequeue();
                finishedWave.jobHandle.Complete();

                ComputeBuffer currentPointBuffer = wavePointBuffers[finishedWave.waveBufferIndex];
                ComputeBuffer currentArgsBuffer = drawArgsBuffers[finishedWave.waveBufferIndex];
                ComputeBuffer currentHitsBuffer = raycastHitsBufferPool[finishedWave.waveBufferIndex];
                NativeArray<RaycastHit> currentResults = finishedWave.results;

                SimpleRaycastHit[] simpleHits = new SimpleRaycastHit[rayCount];
                for (int i = 0; i < rayCount; i++)
                {
                    var hit = currentResults[i];
                    simpleHits[i].p = hit.point;
                    simpleHits[i].normal = hit.normal;
                    simpleHits[i].distance = hit.distance;

                    bool hasHit = hit.collider != null;
                    simpleHits[i].colliderInstanceID = hasHit ? hit.collider.GetInstanceID() : 0;

                    // MODIFIED: Check the tag and assign the correct color
                    if (hasHit && colorMap.ContainsKey(hit.collider.tag))
                    {
                        simpleHits[i].color = colorMap[hit.collider.tag];
                    }
                    else
                    {
                        simpleHits[i].color = defaultColor;
                    }
                }

                currentHitsBuffer.SetData(simpleHits);
                currentPointBuffer.SetCounterValue(0);

                instantiatedComputeShader.SetBuffer(kernelIndex, raycastHitsBufferID, currentHitsBuffer);
                instantiatedComputeShader.SetBuffer(kernelIndex, processedPointsOutBufferID, currentPointBuffer);
                instantiatedComputeShader.SetFloat(propagationSpeedID, propagationSpeed);
                instantiatedComputeShader.SetFloat(timeID, finishedWave.triggerTime);
                instantiatedComputeShader.SetInt(rayCountID, rayCount);

                int threadGroups = Mathf.CeilToInt(rayCount / 64.0f);
                instantiatedComputeShader.Dispatch(kernelIndex, threadGroups, 1, 1);

                // For DrawProcedural, the second argument of the args buffer is the instance count, 
                // which corresponds to the number of points.
                ComputeBuffer.CopyCount(currentPointBuffer, currentArgsBuffer, sizeof(uint));
            }

            if (StarterAssetsInputs.Instance?.GetFireInputDown() ?? false)
            {
                TriggerEcholocation();
            }

            for (int i = 0; i < maxWaves; i++)
            {
                // The property block is how we pass our big data buffer to the Shader Graph
                propertyBlock.SetBuffer(processedPointsBufferID, wavePointBuffers[i]);

                Graphics.DrawMeshInstancedIndirect(
                    pointMesh,
                    0,
                    instantiatedMaterial,
                    new Bounds(Vector3.zero, new Vector3(1000.0f, 1000.0f, 1000.0f)),
                    drawArgsBuffers[i],
                    0,
                    propertyBlock
                );
            }
        }

        public void TriggerEcholocation()
        {
            var currentCommands = commandsPool[nextWaveIndex];
            var currentResults = resultsPool[nextWaveIndex];

            Vector3 origin = transform.position;
            for (int i = 0; i < rayCount; i++)
            {
                currentCommands[i] = new RaycastCommand(origin, UnityEngine.Random.onUnitSphere, QueryParameters.Default, maxDistance);
            }

            var handle = RaycastCommand.ScheduleBatch(currentCommands, currentResults, 1);

            pendingWaves.Enqueue(new PendingWave
            {
                jobHandle = handle,
                waveBufferIndex = nextWaveIndex,
                results = currentResults,
                triggerTime = Time.time
            });

            nextWaveIndex = (nextWaveIndex + 1) % maxWaves;

            JobHandle.ScheduleBatchedJobs();
        }
    }
}