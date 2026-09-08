#version 330 core

in vec2 v_uv;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_frag_pos;
in vec4 v_light_space_pos;
in vec4 v_instance_color;

out vec4 FragColor;

uniform sampler2D u_texture;
uniform sampler2D u_normal_map;
uniform sampler2D u_roughness_map;
uniform sampler2D u_shadow_map;

uniform int u_use_texture;
uniform int u_use_pbr;
uniform vec3 u_view_pos;
uniform vec3 u_light_direction;
uniform vec3 u_light_color;
uniform vec3 u_ambient_color;

float shadow_visibility() {
    vec3 projected = v_light_space_pos.xyz / max(v_light_space_pos.w, 0.0001);
    projected = projected * 0.5 + 0.5;
    if (projected.z <= 0.0 || projected.z >= 1.0 || projected.x <= 0.0 || projected.x >= 1.0 || projected.y <= 0.0 || projected.y >= 1.0) return 1.0;
    vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            lit += projected.z - 0.0015 <= texture(u_shadow_map, projected.xy + vec2(x, y) * texel).r ? 1.0 : 0.0;
    return mix(0.42, 1.0, lit / 9.0);
}

void main() {
    vec3 N = normalize(v_normal);
    float roughness = 0.85;

    if (u_use_pbr != 0) {
        vec3 T = normalize(v_tangent - dot(v_tangent, N) * N);
        vec3 B = cross(N, T);
        mat3 TBN = mat3(T, B, N);
        vec3 n_tex = texture(u_normal_map, v_uv).rgb * 2.0 - 1.0;
        roughness = texture(u_roughness_map, v_uv).r;
        N = normalize(TBN * n_tex);
    }

    vec4 tex_color = u_use_texture != 0 ? texture(u_texture, v_uv) : vec4(1.0);
    vec4 base = tex_color * v_instance_color;

    vec3 L = normalize(-u_light_direction);
    float diff = max(dot(N, L), 0.0);
    vec3 V = normalize(u_view_pos - v_frag_pos);
    vec3 H = normalize(L + V);
    float spec_power = mix(8.0, 128.0, 1.0 - clamp(roughness, 0.05, 0.95));
    float spec = pow(max(dot(N, H), 0.0), spec_power) * (1.0 - roughness * 0.75);

    float shadow = shadow_visibility();
    vec3 ambient = u_ambient_color;
    vec3 diffuse = diff * u_light_color * shadow;
    vec3 specular = spec * u_light_color * shadow * 0.40;

    vec3 lit_color = (ambient + diffuse) * base.rgb + specular;

    // AO analítico de contato na base (igual ao scene.frag original).
    float contact_ao = mix(0.78, 1.0, smoothstep(0.0, 0.70, v_frag_pos.y));
    lit_color *= contact_ao;

    FragColor = vec4(lit_color, base.a);
}
