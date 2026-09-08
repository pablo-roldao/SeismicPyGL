"""
Sistema de partículas para simulação de poeira e fumaça de desabamento.
Utiliza:
- Billboarding esférico (quads perenemente orientados para a câmera)
- Alpha blending configurável (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
- ShaderProgram dedicado (billboard.vert / billboard.frag)
- Textura com gradiente radial suave
"""

import ctypes
import math
import random
import numpy as np
from OpenGL.GL import (
    glEnable, glDisable, glBlendFunc, glDepthMask,
    glBindTexture, glBindVertexArray, glGenBuffers, glBindBuffer, glBufferData, glBufferSubData,
    glDeleteBuffers, glVertexAttribPointer, glEnableVertexAttribArray,
    glVertexAttribDivisor, glDrawArraysInstanced,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_TEXTURE_2D,
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FLOAT, GL_FALSE, GL_TRIANGLES,
)
try:
    from ..core.mesh import Mesh
    from ..core.shader import ShaderProgram
    from ..core.texture import load_texture
except ImportError:
    from mesh import Mesh
    from shader import ShaderProgram
    from texture import load_texture


class ParticleSystem:
    """
    Armazena as partículas ativas em arrays NumPy paralelos (Structure of
    Arrays) em vez de uma lista de objetos, compactados no início dos
    arrays (self._count marca quantos slots estão ativos). update()/draw()
    operam vetorizados sobre a fatia [:self._count] — com até 6000
    partículas simultâneas (colapso de vários prédios), um laço Python por
    partícula media ~5.4ms/frame; vetorizado isso cai para operações NumPy
    em C.
    """

    def __init__(self, max_particles=6000):
        self.max_particles = max_particles
        self._count = 0
        self.pos = np.zeros((max_particles, 3), dtype=np.float32)
        self.vel = np.zeros((max_particles, 3), dtype=np.float32)
        self.size = np.zeros(max_particles, dtype=np.float32)
        self.initial_size = np.zeros(max_particles, dtype=np.float32)
        self.max_size = np.zeros(max_particles, dtype=np.float32)
        self.alpha = np.zeros(max_particles, dtype=np.float32)
        self.life = np.zeros(max_particles, dtype=np.float32)
        self.lifetime = np.ones(max_particles, dtype=np.float32)
        self.color = np.zeros((max_particles, 3), dtype=np.float32)
        self._ambient_dust_budget = 0.0
        self._leaf_budget = 0.0

        # Recursos GPU
        self.mesh = Mesh.create_quad(size=1.0)
        self.instance_vbo = glGenBuffers(1)
        self.instance_stride = 8 * 4  # posição, tamanho, alpha e cor RGB

        # Os dados que mudam por partícula ficam num VBO dinâmico. O quad é
        # compartilhado pela GPU e glDrawArraysInstanced o replica em uma só
        # draw call, inclusive quando há milhares de partículas no colapso.
        glBindVertexArray(self.mesh.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.max_particles * self.instance_stride, None, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, self.instance_stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, self.instance_stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, self.instance_stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(5)
        glVertexAttribDivisor(5, 1)
        glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, self.instance_stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(6)
        glVertexAttribDivisor(6, 1)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        self.texture_id = load_texture("assets/textures/smoke.png")
        self.shader = ShaderProgram.from_files(
            "assets/shaders/billboard.vert",
            "assets/shaders/billboard.frag"
        )

    def _alloc(self, n: int) -> int:
        """Reserva n slots livres no fim da região ativa; retorna quantos couberam."""
        n = min(n, self.max_particles - self._count)
        return max(n, 0)

    def emit(self, position, count=35, spread=1.2, base_speed=1.8):
        """Dispara uma rajada de partículas em torno de position (vetorizado)."""
        n = self._alloc(count)
        if n == 0:
            return
        px, py, pz = position
        sl = slice(self._count, self._count + n)

        angle = np.random.uniform(0, 2 * math.pi, n)
        speed = np.random.uniform(0.4, base_speed, n)
        vx = np.cos(angle) * speed + np.random.uniform(-0.2, 0.2, n)
        vy = np.random.uniform(0.5, base_speed * 1.5, n)
        vz = np.sin(angle) * speed + np.random.uniform(-0.2, 0.2, n)

        # Posição inicial ligeiramente espalhada na base
        offset_r = np.random.uniform(0.0, spread, n)
        offset_a = np.random.uniform(0, 2 * math.pi, n)
        x = px + np.cos(offset_a) * offset_r
        y = py + np.random.uniform(0.0, 0.4, n)
        z = pz + np.sin(offset_a) * offset_r

        initial_size = np.random.uniform(0.6, 1.8, n)
        gray = np.random.uniform(0.70, 0.85, n)

        self.pos[sl] = np.stack([x, y, z], axis=1)
        self.vel[sl] = np.stack([vx, vy, vz], axis=1)
        self.initial_size[sl] = initial_size
        self.size[sl] = initial_size
        self.max_size[sl] = initial_size * np.random.uniform(2.0, 3.5, n)
        self.lifetime[sl] = np.random.uniform(1.5, 3.2, n)
        self.life[sl] = 0.0
        self.alpha[sl] = 1.0
        self.color[sl] = np.stack([gray, gray * 0.95, gray * 0.90], axis=1)

        self._count += n

    def update(self, dt: float):
        """Atualiza a simulação das partículas e descarta as expiradas (vetorizado)."""
        n = self._count
        if n == 0:
            return

        life = self.life[:n] + dt
        t = life / self.lifetime[:n]

        # Movimento e amortecimento
        self.pos[:n] += self.vel[:n] * dt
        decay = 0.95 ** (dt * 60.0)
        self.vel[:n, 0] *= decay
        self.vel[:n, 2] *= decay
        self.vel[:n, 1] = self.vel[:n, 1] * decay + 0.35 * dt

        # Expansão da nuvem de poeira e desvanecimento do alpha
        t_clamped = np.clip(t, 0.0, None)
        self.size[:n] = self.initial_size[:n] + (self.max_size[:n] - self.initial_size[:n]) * np.power(t_clamped, 0.7)
        self.alpha[:n] = np.power(np.clip(1.0 - t, 0.0, None), 1.3)
        self.life[:n] = life

        alive = t < 1.0
        new_count = int(np.count_nonzero(alive))
        if new_count != n:
            for arr in (self.pos, self.vel, self.size, self.initial_size,
                        self.max_size, self.alpha, self.life, self.lifetime, self.color):
                arr[:new_count] = arr[:n][alive]
        self._count = new_count

    def emit_ambient(self, camera_position, forest, dt: float):
        """Pólen e folhas discretos em repouso; continuam no VBO instanciado."""
        self._ambient_dust_budget += dt * 3.0
        self._leaf_budget += dt * 0.9
        cx, cy, cz = camera_position

        while self._ambient_dust_budget >= 1.0 and self._count < self.max_particles:
            self._ambient_dust_budget -= 1.0
            angle = random.uniform(0.0, math.tau)
            radius = random.uniform(4.0, 20.0)
            initial_size = random.uniform(0.05, 0.13)
            i = self._count
            self.pos[i] = (cx + math.cos(angle) * radius, random.uniform(0.35, 3.5), cz + math.sin(angle) * radius)
            self.vel[i] = (random.uniform(-0.18, 0.18), random.uniform(0.08, 0.30), random.uniform(-0.18, 0.18))
            self.initial_size[i] = initial_size
            self.size[i] = initial_size
            self.max_size[i] = initial_size * 1.25
            self.lifetime[i] = random.uniform(4.0, 7.0)
            self.life[i] = 0.0
            self.alpha[i] = 1.0
            self.color[i] = (0.90, 0.82, 0.62)
            self._count += 1

        while self._leaf_budget >= 1.0 and self._count < self.max_particles:
            self._leaf_budget -= 1.0
            nearby = [t for t in forest if math.hypot(t.x - cx, t.z - cz) < 30.0]
            if not nearby:
                continue
            tree = random.choice(nearby)
            color = random.choice(((0.21, 0.38, 0.12), (0.38, 0.29, 0.10), (0.48, 0.36, 0.12)))
            initial_size = random.uniform(0.07, 0.15)
            i = self._count
            self.pos[i] = (
                tree.x + random.uniform(-tree.foliage_radius, tree.foliage_radius),
                tree.trunk_height + tree.foliage_height * random.uniform(0.45, 0.95),
                tree.z + random.uniform(-tree.foliage_radius, tree.foliage_radius),
            )
            self.vel[i] = (random.uniform(-0.45, 0.45), random.uniform(-0.55, -0.18), random.uniform(-0.45, 0.45))
            self.initial_size[i] = initial_size
            self.size[i] = initial_size
            self.max_size[i] = initial_size
            self.lifetime[i] = random.uniform(2.5, 4.5)
            self.life[i] = 0.0
            self.alpha[i] = 1.0
            self.color[i] = color
            self._count += 1

    def draw(self, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        """
        Renderiza todas as partículas com billboarding esférico e alpha blending.
        """
        n = self._count
        if n == 0:
            return

        # Extrai Right e Up em espaço de mundo para orientar os quads.
        cam_right = view_matrix[0, 0:3]
        cam_up = view_matrix[1, 0:3]

        instance_data = np.column_stack((
            self.pos[:n], self.size[:n], self.alpha[:n], self.color[:n]
        )).astype(np.float32)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(False)  # Desabilita escrita no depth buffer para transparência perfeita

        self.shader.use()
        self.shader.set_uniform_mat4("u_view", view_matrix)
        self.shader.set_uniform_mat4("u_projection", projection_matrix)
        self.shader.set_uniform_vec3("u_camera_right", cam_right)
        self.shader.set_uniform_vec3("u_camera_up", cam_up)
        self.shader.set_uniform_int("u_texture", 0)

        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(self.mesh.vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self.mesh.vertex_count, n)
        glBindVertexArray(0)

        self.shader.stop()
        glDepthMask(True)
        glBindTexture(GL_TEXTURE_2D, 0)

    def cleanup(self):
        """Libera malha e shader do sistema de partículas."""
        if self.mesh:
            self.mesh.cleanup()
            self.mesh = None
        if self.instance_vbo is not None:
            glDeleteBuffers(1, [self.instance_vbo])
            self.instance_vbo = None
        if self.shader:
            self.shader.cleanup()
            self.shader = None
