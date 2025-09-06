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
void GetOnePointData_float(uint instanceID, in float currentTime, in float lifeTime, in float3 localVertexPositionOS, in float pointSize, out float3 position, out float alpha, out float4 color)
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

    if (age < 0.0)
    {
        alpha = 0.0;
        return;
    }
    float age_factor = 1.0 - saturate(age / lifeTime);

    alpha = age_factor;
    float size_factor = 0.5 + age_factor;


    float4 pointViewPos = mul(UNITY_MATRIX_V, float4(data.position, 1.0));

    float3 offsetView = float3(localVertexPositionOS.x, localVertexPositionOS.y, 0.0) * pointSize * size_factor;
    pointViewPos.xyz += offsetView;

    float4 finalWorldPos = mul(UNITY_MATRIX_I_V, pointViewPos);
    position = finalWorldPos.xyz;


}

#endif // POINT_DATA_READER_HLSL

