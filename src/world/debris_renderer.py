"""
Renderizador instanciado de escombros (prédios e montanha).

Antes, cada BuildingDebris/RockDebris emitia sua própria sequência de
glActiveTexture/glBindTexture/glUniform*/glDrawArrays — com centenas a
milhares de pedaços de escombro após um terremoto forte, isso vira o maior
gargalo de chamadas OpenGL do frame. Este módulo junta todos os escombros
vivos num único VBO de instância (mesma técnica de `ParticleSystem`,
glDrawArraysInstanced + glVertexAttribDivisor) e desenha todo o escombro de
prédios em 1 draw call e toda a rocha da montanha em outro.
"""

import ctypes
import os
import numpy as np
from OpenGL.GL import (
    glBindTexture, glActiveTexture, glBindVertexArray,
    glGenBuffers, glBindBuffer, glBufferData, glBufferSubData, glDeleteBuffers,
    glVertexAttribPointer, glEnableVertexAttribArray, glVertexAttribDivisor,
    glDrawArraysInstanced,
    GL_TEXTURE_2D, GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3,
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FLOAT, GL_FALSE, GL_TRIANGLES,
)
from ..core.mesh import Mesh
from ..core.shader import ShaderProgram
from ..core.math_utils import to_gl_matrix
from .shared import get_pbr_set


class DebrisRenderer:
    # 16 floats (matriz de modelo, column-major) + 4 floats (cor RGBA) por instância.
    INSTANCE_FLOATS = 20
    INSTANCE_STRIDE = INSTANCE_FLOATS * 4

    def __init__(self, max_instances=4000):
        self.max_instances = max_instances
        self.mesh = Mesh.create_cube()
        self.instance_vbo = glGenBuffers(1)

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        shaders_dir = os.path.join(base_dir, "assets", "shaders")
        self.shader = ShaderProgram.from_files(
            os.path.join(shaders_dir, "debris.vert"),
            os.path.join(shaders_dir, "debris.frag"),
        )
        self.shadow_shader = ShaderProgram.from_files(
            os.path.join(shaders_dir, "debris_shadow.vert"),
            os.path.join(shaders_dir, "shadow.frag"),
        )

        glBindVertexArray(self.mesh.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.max_instances * self.INSTANCE_STRIDE, None, GL_DYNAMIC_DRAW)

        # Matriz de modelo por instância: 4 colunas vec4 nas locations 4-7.
        for col in range(4):
            loc = 4 + col
            glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, self.INSTANCE_STRIDE, ctypes.c_void_p(col * 16))
            glEnableVertexAttribArray(loc)
            glVertexAttribDivisor(loc, 1)

        # Cor por instância na location 8.
        glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, self.INSTANCE_STRIDE, ctypes.c_void_p(64))
        glEnableVertexAttribArray(8)
        glVertexAttribDivisor(8, 1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.concrete_pbr = get_pbr_set("cracked_concrete_02")

        self._building_instances = np.zeros((0, self.INSTANCE_FLOATS), dtype=np.float32)
        self._rock_instances = np.zeros((0, self.INSTANCE_FLOATS), dtype=np.float32)

    def _pack(self, items):
        if not items:
            return np.zeros((0, self.INSTANCE_FLOATS), dtype=np.float32)
        rows = np.empty((len(items), self.INSTANCE_FLOATS), dtype=np.float32)
        for i, (model, color) in enumerate(items):
            rows[i, :16] = to_gl_matrix(model).reshape(-1)
            rows[i, 16:20] = color
        return rows

    def collect(self, buildings):
        """Reúne os escombros liberados de todos os prédios/casas (um único material)."""
        items = [d.instance_data() for b in buildings for d in b.debris if d.released]
        self._building_instances = self._pack(items)

    def collect_rocks(self, mountain):
        """Reúne os blocos de rocha soltos pela montanha (sem textura, cor por instância)."""
        items = [r.instance_data() for r in mountain.debris]
        self._rock_instances = self._pack(items)

    def _upload_and_draw(self, instances):
        count = min(len(instances), self.max_instances)
        if count == 0:
            return
        data = instances[:count]
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindVertexArray(self.mesh.vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self.mesh.vertex_count, count)
        glBindVertexArray(0)

    def draw_shadow(self, light_space_matrix):
        if len(self._building_instances) == 0 and len(self._rock_instances) == 0:
            return
        self.shadow_shader.use()
        self.shadow_shader.set_uniform_mat4("u_light_space_matrix", light_space_matrix)
        self._upload_and_draw(self._building_instances)
        self._upload_and_draw(self._rock_instances)
        self.shadow_shader.stop()

    def draw(self, view_matrix, proj_matrix, light_space_matrix, shadow_texture, view_pos,
             light_direction=(-0.4, -1.0, -0.3), light_color=(1.0, 0.98, 0.92), ambient_color=(0.35, 0.38, 0.42)):
        if len(self._building_instances) == 0 and len(self._rock_instances) == 0:
            return

        self.shader.use()
        self.shader.set_uniform_mat4("u_view", view_matrix)
        self.shader.set_uniform_mat4("u_projection", proj_matrix)
        self.shader.set_uniform_mat4("u_light_space_matrix", light_space_matrix)
        self.shader.set_uniform_vec3("u_light_direction", light_direction)
        self.shader.set_uniform_vec3("u_light_color", light_color)
        self.shader.set_uniform_vec3("u_ambient_color", ambient_color)
        self.shader.set_uniform_vec3("u_view_pos", view_pos)

        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, shadow_texture)
        self.shader.set_uniform_int("u_shadow_map", 3)

        # Escombro de prédio: material PBR único compartilhado por todos os prédios.
        if len(self._building_instances) > 0:
            self.shader.set_uniform_int("u_use_texture", 1)
            self.shader.set_uniform_int("u_use_pbr", 1)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.concrete_pbr["albedo"])
            self.shader.set_uniform_int("u_texture", 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.concrete_pbr["normal"])
            self.shader.set_uniform_int("u_normal_map", 1)
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, self.concrete_pbr["roughness"])
            self.shader.set_uniform_int("u_roughness_map", 2)
            self._upload_and_draw(self._building_instances)

        # Rocha da montanha: sem textura, cor sólida por instância.
        if len(self._rock_instances) > 0:
            self.shader.set_uniform_int("u_use_texture", 0)
            self.shader.set_uniform_int("u_use_pbr", 0)
            self._upload_and_draw(self._rock_instances)

        self.shader.stop()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def cleanup(self):
        if self.mesh:
            self.mesh.cleanup()
            self.mesh = None
        if self.instance_vbo is not None:
            glDeleteBuffers(1, [self.instance_vbo])
            self.instance_vbo = None
        if self.shader:
            self.shader.cleanup()
            self.shader = None
        if self.shadow_shader:
            self.shadow_shader.cleanup()
            self.shadow_shader = None
