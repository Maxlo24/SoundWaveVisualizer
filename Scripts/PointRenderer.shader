Shader "Custom/PointRenderer"
{
    Properties
    {
        _Color ("Point Color", Color) = (1, 1, 1, 1)
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" }
        LOD 100
        Cull Off
        ZWrite Off
        Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5 // Required for StructuredBuffers

            #include "UnityCG.cginc"

            // This must match the PointData struct in C# and the compute shader
            struct PointData {
                float3 position;
                float3 normal;
                float startTime;
            };

            struct appdata
            {
                float4 vertex : POSITION;
                uint instanceID : SV_InstanceID;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float4 color : COLOR;
            };
            
            // The buffer containing all point data, set from the C# script
            StructuredBuffer<PointData> _PointsBuffer;
            
            // Properties
            float4 _Color;
            float _LifeTime;

            v2f vert (appdata v)
            {
                v2f o;

                PointData p = _PointsBuffer[v.instanceID];

                float age = _Time.y - p.startTime;

                if (age < 0) {
                    o.vertex = float4(0, 0, 0, 0);
                    o.color = float4(0, 0, 0, 0);
                    return o;
                }

                float fade = 1.0 - saturate(age / _LifeTime);

                if (fade <= 0)
                {
                    o.vertex = float4(0, 0, 0, 0);
                    o.color = float4(0, 0, 0, 0);
                }
                else
                {
                    float3 worldPos = p.position + v.vertex.xyz * 0.05;
                    o.vertex = UnityObjectToClipPos(float4(worldPos, 1.0));
                    o.color = float4(_Color.rgb, _Color.a * fade);
                }

                return o;
            }


            fixed4 frag (v2f i) : SV_Target
            {
                // Apply the calculated color and alpha
                return i.color;
            }
            ENDCG
        }
    }
}