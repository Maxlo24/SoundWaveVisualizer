using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using UnityEngine.Rendering;
using System.Runtime.InteropServices;
using UnityEngine.InputSystem;

// Define the structure that will be passed to the GPU.
// This must match the PointData struct in the compute and rendering shaders.
// It's crucial that the memory layout is sequential.

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
        public float pointLifetime = 3.0f; // How long points last before fading out

        [Header("References")]
        public ComputeShader computeShader;
        public Mesh pointMesh; // A simple mesh to represent each point (e.g., a small quad)
        public Material pointMaterial; // Material using the PointRenderer shader

        [Header("Wave Settings")]
        public float propagationSpeed = 50f; // meters per second

        // NativeArrays for the C# Job System (Steps 2 & 3)
        private NativeArray<RaycastCommand> commands;
        private NativeArray<RaycastHit> results;
        private JobHandle raycastHandle;

        // Compute Buffers for GPU data transfer (Steps 4, 6, 7)
        private ComputeBuffer raycastHitsBuffer;     // Input for Compute Shader
        private ComputeBuffer processedPointsBuffer; // Output of Compute Shader / Input for Renderer
        private ComputeBuffer drawArgsBuffer;        // For indirect drawing arguments

        // Shader property IDs for performance
        private int processedPointsBufferID = Shader.PropertyToID("_PointsBuffer");
        private int propagationSpeedID = Shader.PropertyToID("_PropagationSpeed");
        private int timeID = Shader.PropertyToID("_Time");
        private int rayCountID = Shader.PropertyToID("_RayCount");
        private int lifetimeID = Shader.PropertyToID("_LifeTime");

        private int kernelIndex;

        void Start()
        {
            // --- Initialization ---
            // 1. Allocate memory that is not managed by the garbage collector.
            commands = new NativeArray<RaycastCommand>(rayCount, Allocator.Persistent);
            results = new NativeArray<RaycastHit>(rayCount, Allocator.Persistent);

            // 2. Initialize the Compute Buffers.
            // We use a simplified RaycastHit struct on the GPU side. Unity's RaycastHit is complex,
            // but its data is laid out sequentially in a way that the essential parts (point, normal, colliderID)
            // can be read by the compute shader.
            int simpleRaycastHitStride = sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) + sizeof(int);
            raycastHitsBuffer = new ComputeBuffer(rayCount, simpleRaycastHitStride, ComputeBufferType.Default);

            processedPointsBuffer = new ComputeBuffer(rayCount, sizeof(float) * 7, ComputeBufferType.Append);

            // Indirect draw arguments: [index count per instance, instance count, start index, base vertex, start instance]
            drawArgsBuffer = new ComputeBuffer(1, 5 * sizeof(uint), ComputeBufferType.IndirectArguments);
            uint[] args = new uint[5] { 0, 0, 0, 0, 0 };
            args[0] = pointMesh.GetIndexCount(0);
            args[1] = (uint)rayCount;
            args[2] = pointMesh.GetIndexStart(0);
            args[3] = pointMesh.GetBaseVertex(0);
            drawArgsBuffer.SetData(args);

            // 3. Link buffers and properties to shaders.
            kernelIndex = computeShader.FindKernel("GeneratePoints");
            computeShader.SetBuffer(kernelIndex, "_RaycastHitsBuffer", raycastHitsBuffer);
            computeShader.SetBuffer(kernelIndex, "_ProcessedPointsBuffer", processedPointsBuffer);
            pointMaterial.SetBuffer(processedPointsBufferID, processedPointsBuffer);
            pointMaterial.SetFloat(lifetimeID, pointLifetime);
        }

        void OnDestroy()
        {
            // --- Cleanup ---
            // Always release native collections and buffers to avoid memory leaks.
            if (commands.IsCreated) commands.Dispose();
            if (results.IsCreated) results.Dispose();

            raycastHitsBuffer?.Release();
            processedPointsBuffer?.Release();
            drawArgsBuffer?.Release();
        }

        void Update()
        {
            // Step 1: Sound Trigger
            // For demonstration, trigger with a key press.

            if (StarterAssetsInputs.Instance != null)
            {
                if (StarterAssetsInputs.Instance.GetFireInputDown())
                {
                    TriggerEcholocation();
                }
            }

            // Step 7: GPU Rendering
            // This is called every frame to draw the points currently alive on the GPU.
            Graphics.DrawMeshInstancedIndirect(
                pointMesh,
                0,
                pointMaterial,
                new Bounds(Vector3.zero, new Vector3(1000.0f, 1000.0f, 1000.0f)),
                drawArgsBuffer,
                0,
                null,
                ShadowCastingMode.Off,
                false
            );
        }

        public void TriggerEcholocation()
        {
            Debug.Log("Start ecolocation");
            // --- Frame Logic ---
            // Ensure the previous job is complete before starting a new one.
            raycastHandle.Complete();

            // Step 2: Job Scheduling
            // Populate the commands array with rays in a spherical pattern.
            Vector3 origin = transform.position;

            for (int i = 0; i < rayCount; i++)
            {
                Vector3 direction = UnityEngine.Random.onUnitSphere;
                commands[i] = new RaycastCommand(origin, direction, QueryParameters.Default, maxDistance);
            }

            // Step 3: Physics Job Execution
            raycastHandle = RaycastCommand.ScheduleBatch(commands, results, 1, default);

            // This must be called before we can use the results.
            raycastHandle.Complete();

            // Step 4: CPU-GPU Data Transfer
            // Copy the raw hit data to the GPU buffer.
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
            processedPointsBuffer.SetCounterValue(0); // Reset the append buffer counter

            // Step 5: Compute Shader Dispatch
            computeShader.SetFloat(propagationSpeedID, propagationSpeed);
            computeShader.SetFloat(timeID, Time.time);
            computeShader.SetInt(rayCountID, rayCount);

            // Tell the GPU to run the compute shader.
            // The number of thread groups is calculated to cover all rays.
            int threadGroups = Mathf.CeilToInt(rayCount / 64.0f);
            computeShader.Dispatch(kernelIndex, threadGroups, 1, 1);

            // Update the instance count in the indirect draw buffer from the append buffer's counter.
            ComputeBuffer.CopyCount(processedPointsBuffer, drawArgsBuffer, sizeof(uint));
        }
    }
}