// This file acts as a bridge between the C# script's compute buffer
// and the Shader Graph's Custom Function node.

#ifndef POINT_DATA_READER_HLSL
#define POINT_DATA_READER_HLSL

// This struct definition MUST EXACTLY MATCH the 'PointData' struct in EcholocationController.cs.
// C# 'Color' becomes a 'float4' in HLSL.
// C# 'Vector3' becomes a 'float3' in HLSL.
struct PointData {
    float3 position;
    float startTime;
    float4 color;
};

// This is the buffer that our C# script fills and passes to the shader via a MaterialPropertyBlock.
// The name '_PointsBuffer' MUST EXACTLY MATCH the name used in the Shader.PropertyToID call in C#.
StructuredBuffer<PointData> _PointsBuffer;

// This is the function that the Shader Graph Custom Function node will call.
// It takes the ID of the current instance (or vertex) being drawn and returns the
// corresponding data from our buffer.
void GetOnePointData_float(uint instanceID, in float currentTime, in float lifeTime, in float3 localVertexPositionOS, in float pointSize,in float4x4 cameraToWorld, out float3 position, out float alpha, out float4 color)
{
    //PointData data = _PointsBuffer[instanceID];
    //position = data.position;
    //startTime = data.startTime;
    //color = data.color;

     // Step 1: Fetch the unique data for this instance from the buffer.
    PointData data = _PointsBuffer[instanceID];

    // Step 2: Pass out the simple data directly.
    color = data.color;

    // Step 3: Perform the alpha calculation.
    float age = currentTime - data.startTime;

    // If the point's start time is in the future, its age is negative.
    // In this case, its alpha should be 0 so it's not visible yet.
    if (age < 0.0)
    {
        alpha = 0.0;
        return;
    }

    // Calculate the fade, clamp it between 0 and 1, and invert it.
    // (saturate is a cheap way to do a min(max(x, 0), 1) clamp)
    alpha = 1.0 - saturate(age / lifeTime);


    float3 cameraUpWS = cameraToWorld[1].xyz;
    float3 cameraRightWS = cameraToWorld[0].xyz;

    float3 offset = (cameraRightWS * localVertexPositionOS.x + cameraUpWS * localVertexPositionOS.y) * pointSize;

    position = data.position + offset;
}

#endif // POINT_DATA_READER_HLSL

