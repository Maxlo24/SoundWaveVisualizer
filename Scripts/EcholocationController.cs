using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using UnityEngine.Rendering;
using System.Runtime.InteropServices;

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
        [Header("Configuration")]
        public int rayCount = 50000;
        public float maxDistance = 100f;
        public float pointLifetime = 3.0f;
        [Tooltip("The maximum number of waves to stack before overwriting the oldest.")]
        public int maxWaves = 10;

        [Header("References")]
        public ComputeShader computeShader;
        public Mesh pointMesh;
        public Material pointMaterial;

        [Header("Wave Settings")]
        public float propagationSpeed = 50f;

        private NativeArray<RaycastCommand> commands;
        private NativeArray<RaycastHit> results;
        private JobHandle raycastHandle;

        private ComputeBuffer raycastHitsBuffer;
        private ComputeBuffer processedPointsBuffer;
        private ComputeBuffer drawArgsBuffer;

        private Material instantiatedMaterial;
        private ComputeShader instantiatedComputeShader;

        // --- NEW ---
        // Tracks which wave "slot" we are currently writing to.
        private int currentWaveIndex = 0;
        // The total number of points currently alive in the VRAM buffer.
        private int totalPointsInVRAM = 0;
        // We need a CPU-side copy of the args to update them.
        private uint[] drawArgs;


        // Shader property IDs
        private static readonly int
            processedPointsBufferID = Shader.PropertyToID("_PointsBuffer"),
            propagationSpeedID = Shader.PropertyToID("_PropagationSpeed"),
            timeID = Shader.PropertyToID("_Time"),
            rayCountID = Shader.PropertyToID("_RayCount"),
            lifetimeID = Shader.PropertyToID("_LifeTime"),
            pointOffsetID = Shader.PropertyToID("_PointOffset"); // New ID for the offset

        private int kernelIndex;

        void Start()
        {
            instantiatedMaterial = new Material(pointMaterial);
            instantiatedComputeShader = Instantiate(computeShader);

            commands = new NativeArray<RaycastCommand>(rayCount, Allocator.Persistent);
            results = new NativeArray<RaycastHit>(rayCount, Allocator.Persistent);

            int simpleRaycastHitStride = sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) + sizeof(int);
            raycastHitsBuffer = new ComputeBuffer(rayCount, simpleRaycastHitStride, ComputeBufferType.Default);

            // --- MODIFIED ---
            // The buffer is no longer an Append buffer. It's a default, read/write buffer.
            int maxPointCount = rayCount * maxWaves;
            processedPointsBuffer = new ComputeBuffer(maxPointCount, sizeof(float) * 7, ComputeBufferType.Default);

            drawArgsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);
            drawArgs = new uint[5] { 0, 0, 0, 0, 0 };
            drawArgs[0] = pointMesh.GetIndexCount(0);
            drawArgs[1] = 0; // Start with 0 points to draw.
            drawArgs[2] = pointMesh.GetIndexStart(0);
            drawArgs[3] = pointMesh.GetBaseVertex(0);
            drawArgsBuffer.SetData(drawArgs);

            kernelIndex = instantiatedComputeShader.FindKernel("GeneratePoints");
            instantiatedComputeShader.SetBuffer(kernelIndex, "_RaycastHitsBuffer", raycastHitsBuffer);
            instantiatedComputeShader.SetBuffer(kernelIndex, "_ProcessedPointsBuffer", processedPointsBuffer);

            instantiatedMaterial.SetBuffer(processedPointsBufferID, processedPointsBuffer);
            instantiatedMaterial.SetFloat(lifetimeID, pointLifetime);
        }

        void OnDestroy()
        {
            if (commands.IsCreated) commands.Dispose();
            if (results.IsCreated) results.Dispose();
            raycastHitsBuffer?.Release();
            processedPointsBuffer?.Release();
            drawArgsBuffer?.Release();

            if (instantiatedMaterial != null) Destroy(instantiatedMaterial);
            if (instantiatedComputeShader != null) Destroy(instantiatedComputeShader);
        }

        void Update()
        {
            if (StarterAssetsInputs.Instance?.GetFireInputDown() ?? false)
            {
                TriggerEcholocation();
            }

            Graphics.DrawMeshInstancedIndirect(
                pointMesh,
                0,
                instantiatedMaterial,
                new Bounds(Vector3.zero, new Vector3(1000.0f, 1000.0f, 1000.0f)),
                drawArgsBuffer
            );
        }

        public void TriggerEcholocation()
        {
            raycastHandle.Complete();

            // --- MODIFIED: Circular Buffer Logic ---
            // 1. Calculate the offset for the current wave.
            int pointOffset = currentWaveIndex * rayCount;
            Debug.Log($"Start Echolocation Wave: {currentWaveIndex + 1}/{maxWaves}. Writing at offset: {pointOffset}");

            // 2. Schedule physics jobs.
            Vector3 origin = transform.position;
            for (int i = 0; i < rayCount; i++)
            {
                commands[i] = new RaycastCommand(origin, UnityEngine.Random.onUnitSphere, QueryParameters.Default, maxDistance);
            }
            raycastHandle = RaycastCommand.ScheduleBatch(commands, results, 1, default);
            raycastHandle.Complete();

            // 3. Copy raycast hits to a temporary buffer.
            SimpleRaycastHit[] simpleHits = new SimpleRaycastHit[rayCount];
            for (int i = 0; i < rayCount; i++)
            {
                var hit = results[i];
                simpleHits[i].p = hit.point;
                simpleHits[i].normal = hit.normal;
                simpleHits[i].distance = hit.distance;
                simpleHits[i].colliderInstanceID = hit.collider != null ? hit.collider.GetInstanceID() : 0;
            }
            raycastHitsBuffer.SetData(simpleHits);

            // 4. Dispatch the compute shader.
            instantiatedComputeShader.SetFloat(propagationSpeedID, propagationSpeed);
            instantiatedComputeShader.SetFloat(timeID, Time.time);
            instantiatedComputeShader.SetInt(rayCountID, rayCount);
            instantiatedComputeShader.SetInt(pointOffsetID, pointOffset);

            int threadGroups = Mathf.CeilToInt(rayCount / 64.0f);
            instantiatedComputeShader.Dispatch(kernelIndex, threadGroups, 1, 1);

            // --- MODIFIED: Manually update the draw argument buffer.
            // 5. Update the total number of points to draw.
            // We increase the count until the buffer is full, then it stays at max.
            if (totalPointsInVRAM < rayCount * maxWaves)
            {
                totalPointsInVRAM += rayCount;
            }
            drawArgs[1] = (uint)totalPointsInVRAM;
            drawArgsBuffer.SetData(drawArgs);

            // --- NEW ---
            // 6. Move to the next "slot" in the circular buffer for the next wave.
            currentWaveIndex = (currentWaveIndex + 1) % maxWaves;
        }
    }
}