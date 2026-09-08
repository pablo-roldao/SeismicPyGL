#version 330 core

layout (location = 0) in vec3 a_position;
layout (location = 4) in vec4 a_instance_model_0;
layout (location = 5) in vec4 a_instance_model_1;
layout (location = 6) in vec4 a_instance_model_2;
layout (location = 7) in vec4 a_instance_model_3;

uniform mat4 u_light_space_matrix;

void main() {
    mat4 model = mat4(a_instance_model_0, a_instance_model_1, a_instance_model_2, a_instance_model_3);
    gl_Position = u_light_space_matrix * model * vec4(a_position, 1.0);
}
