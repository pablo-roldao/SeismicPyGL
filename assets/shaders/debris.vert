#version 330 core

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec2 a_uv;
layout (location = 2) in vec3 a_normal;
layout (location = 3) in vec3 a_tangent;
layout (location = 4) in vec4 a_instance_model_0;
layout (location = 5) in vec4 a_instance_model_1;
layout (location = 6) in vec4 a_instance_model_2;
layout (location = 7) in vec4 a_instance_model_3;
layout (location = 8) in vec4 a_instance_color;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_light_space_matrix;

out vec2 v_uv;
out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_frag_pos;
out vec4 v_light_space_pos;
out vec4 v_instance_color;

void main() {
    mat4 model = mat4(a_instance_model_0, a_instance_model_1, a_instance_model_2, a_instance_model_3);
    vec4 world_pos = model * vec4(a_position, 1.0);
    v_frag_pos = world_pos.xyz;

    mat3 normal_matrix = transpose(inverse(mat3(model)));
    v_normal = normalize(normal_matrix * a_normal);
    v_tangent = normalize(normal_matrix * a_tangent);
    v_uv = a_uv;
    v_light_space_pos = u_light_space_matrix * world_pos;
    v_instance_color = a_instance_color;

    gl_Position = u_projection * u_view * world_pos;
}
