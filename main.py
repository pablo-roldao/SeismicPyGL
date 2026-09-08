"""
SeismicPyGL — Simulador 3D de Terremotos em Python com Pygame e PyOpenGL.
Pipeline programável moderno (GLSL 3.3 Core):
- Câmera em primeira pessoa com Trauma Screen Shake via Ruído de Perlin
- Deformação sísmica circular calculada diretamente na GPU (ground.vert)
- Colapso dinâmico de edifícios (afundamento Y, rotação e escombros)
- Sistema de partículas para fumaça/poeira com Billboarding e Alpha Blending
- Iluminação direcional (Phong / produto escalar) e texturas via VBO/VAO
- HUD 2D ortográfico isolado do Depth Test com Escala Richter e FPS estável a 60 FPS
"""

import math
import os
import platform
import sys


def _running_under_wsl() -> bool:
    """Detecta WSL de verdade, em vez de assumir que todo Linux é WSL."""
    if not sys.platform.startswith("linux"):
        return False
    if "microsoft" in platform.uname().release.lower():
        return True
    return os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")


def _has_nvidia_gpu() -> bool:
    """Indício leve (sem subprocesso) de que o driver proprietário NVIDIA está carregado."""
    return sys.platform.startswith("linux") and os.path.exists("/proc/driver/nvidia/version")


# No WSL a GPU dedicada só é acessível via tradução D3D12 do Mesa; em Linux
# nativo esse hack é desnecessário (e pode até forçar um caminho pior que o
# driver nativo/GLVND já escolheria sozinho), então só se aplica sob WSL.
if _running_under_wsl():
    if "GALLIUM_DRIVER" not in os.environ:
        os.environ["GALLIUM_DRIVER"] = "d3d12"
    if "MESA_D3D12_DEFAULT_ADAPTER_NAME" not in os.environ:
        os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = "NVIDIA"
    if "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "glx"
    if "/usr/lib/wsl/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
elif sys.platform.startswith("linux"):
    if _has_nvidia_gpu():
        # Sob um compositor Wayland (comum em notebooks híbridos com
        # Intel+NVIDIA), o SDL costuma escolher o caminho EGL nativo do
        # Wayland para criar o contexto OpenGL, que não consulta a seleção
        # de GPU do GLVND (__GLX_VENDOR_LIBRARY_NAME) — a GPU dedicada pode
        # nunca ser usada mesmo estando disponível e configurada. O PRIME
        # render offload da própria NVIDIA resolve isso sem trocar de
        # backend de janela (ao contrário do SEISMICPYGL_FORCE_X11 abaixo):
        # verificado nesta máquina (Hyprland/Omarchy, Intel Alder Lake +
        # RTX 3050) que "[Hardware 3D] GPU:" passa a mostrar a RTX 3050,
        # de forma estável, mantendo o caminho Wayland/EGL nativo.
        os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

    if os.environ.get("SEISMICPYGL_FORCE_X11") == "1":
        # Último recurso, caso o PRIME render offload acima não baste em
        # alguma outra combinação de compositor/driver: força XWayland/GLX,
        # que sempre respeita a seleção de GPU do GLVND. Não é ativado por
        # padrão porque, em ao menos um ambiente Wayland testado, forçar
        # SDL_VIDEODRIVER=x11 quebrou a criação do contexto OpenGL. Use
        # `SEISMICPYGL_FORCE_X11=1 python main.py` para testar manualmente
        # se a GPU dedicada ainda não está sendo usada (veja a linha
        # "[Hardware 3D] GPU:" impressa no console ao iniciar).
        if "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "x11"

import pygame
from pygame.locals import (
    DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE, K_SPACE,
    K_1, K_2, K_3, K_4, K_5, K_r, MOUSEWHEEL,
)
from OpenGL.GL import (
    glEnable, glClearColor, glClear, glBindTexture, glActiveTexture, glGetString,
    GL_DEPTH_TEST, GL_MULTISAMPLE, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_TEXTURE_2D,
    GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE6, GL_RENDERER, GL_VENDOR, GL_VERSION,
)

from src.core import (
    perspective, look_at, ortho, FreeCamera,
    ShaderProgram, check_gl_error, cleanup_textures, reset_texture_cache,
    get_fallback_count
)
from src.simulation import EarthquakeSimulator, ParticleSystem
from src.rendering import ShadowMap, Sky, HUD
from src.world import (
    Ground, generate_village, Mountain, generate_forest,
    get_concrete_texture, Street, DebrisRenderer, reset_shared_resources
)

# 4K é opcional para não sacrificar 60 FPS em monitores/GPUs menores.
# Ex.: SEISMICPYGL_RESOLUTION=4k python main.py
_resolution = os.environ.get("SEISMICPYGL_RESOLUTION", "1080p").lower()
WINDOW_SIZE = (3840, 2160) if _resolution in {"4k", "2160p", "3840x2160"} else (1920, 1080)
ASPECT_RATIO = WINDOW_SIZE[0] / WINDOW_SIZE[1]


def init_opengl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glClearColor(0.62, 0.78, 0.94, 1.0)  # Cor de céu aberto
    check_gl_error("init_opengl")


def _in_view(obj_x, obj_z, cam_x, cam_z, forward_x, forward_z, cos_threshold, near_radius=6.0):
    """
    Culling barato por ângulo (não é frustum culling completo): descarta
    objetos claramente fora do campo de visão da câmera para poupar as
    chamadas OpenGL de desenhá-los. Objetos muito próximos nunca são
    cortados, para não sumirem se a câmera girar rápido.
    """
    dx = obj_x - cam_x
    dz = obj_z - cam_z
    dist = math.hypot(dx, dz)
    if dist < near_radius:
        return True
    dot = (dx / dist) * forward_x + (dz / dist) * forward_z
    return dot > cos_threshold


def main():
    pygame.init()
    # A GPU faz a suavização das bordas; é barato e melhora bastante a leitura
    # das silhuetas sem precisar aumentar a quantidade de objetos da cena.
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("SeismicPyGL — Simulador 3D de Terremotos (GLSL)")

    # Captura o cursor do mouse para visão em primeira pessoa
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.mouse.get_rel()

    init_opengl()

    try:
        renderer = glGetString(GL_RENDERER).decode()
        vendor = glGetString(GL_VENDOR).decode()
        version = glGetString(GL_VERSION).decode()
        print(f"[Hardware 3D] GPU:    {renderer}")
        print(f"[Hardware 3D] Vendor: {vendor}")
        print(f"[Hardware 3D] Driver: {version}")
    except Exception as e:
        print(f"[Hardware 3D] Info GPU: {e}")

    clock = pygame.time.Clock()
    
    # Enquadra o centro da vila já no primeiro frame.
    camera = FreeCamera(position=(0.0, 5.0, 36.0), yaw=-90.0, pitch=-7.0)

    # Entidades do mundo 3D
    earthquake = EarthquakeSimulator()
    # Grade mais densa: a deformação continua inteiramente no vertex shader.
    ground = Ground(size=140.0, divisions=220)
    
    buildings, houses, streets, lamp_posts = generate_village(center=(0.0, 0.0), building_count=9, house_count=24)
    all_buildings = buildings + houses

    mountain_center = (48.0, 48.0)
    mountain = Mountain(
        x=mountain_center[0], z=mountain_center[1], base_radius=14.0,
        height=26.0, bands=20, slices=72,
    )
    forest = generate_forest(
        count=120,
        center=(0.0, 0.0),
        radius_range=(18.0, 55.0),
        avoid_zones=[(0.0, 0.0, 16.0), (48.0, 48.0, 16.0)],
        avoid_buildings=all_buildings,
        avoid_streets=streets,
    )

    # Shaders e Sistemas de Efeitos
    scene_shader = ShaderProgram.from_files(
        "assets/shaders/scene.vert",
        "assets/shaders/scene.frag"
    )
    particle_system = ParticleSystem(max_particles=6000)
    # Desenha todos os escombros de prédios/montanha em 1-2 draw calls
    # instanciados, em vez de um glDrawArrays por pedaço de escombro.
    debris_renderer = DebrisRenderer(max_instances=4000)
    hud = HUD(WINDOW_SIZE[0], WINDOW_SIZE[1])
    shadow_map = ShadowMap(size=1024)
    sky = Sky()

    fallback_count = get_fallback_count()
    if fallback_count > 0:
        print(f"\033[33m[Aviso]\033[0m {fallback_count} textura(s) usando "
              f"fallback procedural — veja os avisos acima ou confira "
              f"assets/textures/pbr/.")

    # Luz direcional estável para a depth map: uma única passagem extra por frame.
    light_view = look_at((42.0, 65.0, 36.0), (0.0, 0.0, 0.0))
    light_projection = ortho(-72.0, 72.0, -72.0, 72.0, 1.0, 150.0)
    light_space_matrix = light_projection @ light_view

    # Mapeamento de magnitudes Richter e trauma para as teclas 1 a 5
    magnitudes = {K_1: 3.0, K_2: 4.5, K_3: 6.0, K_4: 7.2, K_5: 8.5}
    trauma_map = {K_1: 0.25, K_2: 0.45, K_3: 0.65, K_4: 0.95, K_5: 1.0}

    running = True
    elapsed_time = 0.0
    first_frame = True

    while running:
        dt = clock.tick(60) / 1000.0
        # Limita dt anômalo para estabilidade numérica
        dt = min(dt, 0.05)
        elapsed_time += dt

        # -------------------------------------------------------------------
        # Processamento de Eventos e Teclado
        # -------------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

                elif event.key == K_SPACE:
                    # Terremoto aleatório com intensidade moderada
                    earthquake.trigger(current_time=elapsed_time, magnitude=5.5)
                    camera.add_trauma(0.70)

                elif event.key in magnitudes:
                    # Teclas 1 a 5: magnitudes da Escala Richter calibradas
                    mag = magnitudes[event.key]
                    t_amt = trauma_map[event.key]
                    earthquake.trigger(current_time=elapsed_time, magnitude=mag)
                    camera.add_trauma(t_amt)

                elif event.key == K_r:
                    # Reset geral do cenário
                    for b in all_buildings:
                        b.reset()
                    for tree in forest:
                        tree.reset()
                    for lp in lamp_posts:
                        lp.reset()
                    mountain.debris.clear()
                    earthquake.reset()
                    camera.reset_view()

            elif event.type == MOUSEWHEEL:
                camera.zoom(event.y)

        # Atualização da Câmera (movimento e screen shake)
        rel_x, rel_y = pygame.mouse.get_rel()
        if first_frame:
            first_frame = False
        else:
            camera.process_mouse(rel_x, rel_y)
            
        camera.process_keyboard(dt, buildings=all_buildings, trees=forest,
                                 mountain=mountain, light_poles=lamp_posts)
        camera.update_trauma(dt)

        # Atualização física dos objetos
        for b in all_buildings:
            b.update(earthquake, elapsed_time, dt, particle_system=particle_system)
        for tree in forest:
            tree.update(earthquake, elapsed_time, dt)
        for lp in lamp_posts:
            lp.update(earthquake, elapsed_time, dt, particle_system=particle_system)
        mountain.update(earthquake, elapsed_time, dt)
        particle_system.update(dt)
        if not earthquake.active:
            particle_system.emit_ambient((camera.x, camera.y, camera.z), forest, dt)

        # Reúne os escombros vivos deste frame para o desenho instanciado.
        debris_renderer.collect(all_buildings)
        debris_renderer.collect_rocks(mountain)

        # Vetor de visão da câmera (para o culling barato de árvores/postes abaixo).
        fwd_x, _, fwd_z = camera.forward_vector()
        fwd_len = math.hypot(fwd_x, fwd_z) or 1.0
        fwd_x, fwd_z = fwd_x / fwd_len, fwd_z / fwd_len
        cos_threshold = math.cos(math.radians(camera.fov * 0.5 + 30.0))

        # -------------------------------------------------------------------
        # Renderização 3D (Pipeline Programável)
        # -------------------------------------------------------------------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view_matrix = camera.get_view_matrix()
        proj_matrix = perspective(camera.fov, ASPECT_RATIO, 0.1, 1000.0)

        # Passagem de profundidade: objetos estáticos/dinâmicos projetam sombra.
        # shadow_pass=True pula todo o bind de textura/uniforms PBR, já que o
        # shader de sombra só lê posição + u_model (fragment shader vazio).
        shadow_map.begin(light_space_matrix)
        for s in streets:
            s.draw(shadow_map.shader, earthquake, elapsed_time, shadow_pass=True)
        for b in all_buildings:
            b.draw(shadow_map.shader, earthquake, elapsed_time, shadow_pass=True)
        mountain.draw(shadow_map.shader, shadow_pass=True)
        for tree in forest:
            tree.draw(shadow_map.shader, earthquake, elapsed_time, shadow_pass=True)
        for lp in lamp_posts:
            lp.draw(shadow_map.shader, earthquake, elapsed_time, shadow_pass=True)
        debris_renderer.draw_shadow(light_space_matrix)
        shadow_map.end(*WINDOW_SIZE)

        sky.draw(view_matrix, proj_matrix, elapsed_time)

        # 1. Chão deformável com onda sísmica na GPU e texturas PBR (sparse_grass + cracked_concrete_02)
        ground.draw(earthquake, elapsed_time, view_matrix, proj_matrix, light_space_matrix,
                    shadow_map.texture, all_buildings, camera=camera)

        # 2. Objetos da Cena (Prédios, Casas, Montanha, Árvores, Postes, Ruas) com PBR completo
        scene_shader.use()
        scene_shader.set_uniform_mat4("u_view", view_matrix)
        scene_shader.set_uniform_mat4("u_projection", proj_matrix)
        scene_shader.set_uniform_mat4("u_light_space_matrix", light_space_matrix)
        scene_shader.set_uniform_vec3("u_light_direction", (-0.4, -1.0, -0.3))
        scene_shader.set_uniform_vec3("u_light_color", (1.0, 0.98, 0.92))
        scene_shader.set_uniform_vec3("u_ambient_color", (0.35, 0.38, 0.42))
        scene_shader.set_uniform_vec3("u_view_pos", (camera.x, camera.y, camera.z))

        glActiveTexture(GL_TEXTURE6)
        glBindTexture(GL_TEXTURE_2D, shadow_map.texture)
        scene_shader.set_uniform_int("u_shadow_map", 6)
        glActiveTexture(GL_TEXTURE0)

        crack_intensity = earthquake.get_crack_intensity()
        scene_shader.set_uniform_float("u_crack_intensity", crack_intensity)
        scene_shader.set_uniform_vec2("u_epicenter", earthquake.epicenter)
        scene_shader.set_uniform_float("u_spatial_falloff", earthquake.spatial_falloff)

        # Ruas (asfalto PBR clean_asphalt com placas ondulantes)
        for s in streets:
            s.draw(scene_shader, earthquake, elapsed_time)
        scene_shader.set_uniform_int("u_is_street", 0)

        # Postes de iluminação pública (PBR metal_plate_02) — só os que estão
        # aproximadamente no campo de visão da câmera.
        for lp in lamp_posts:
            if _in_view(lp.x, lp.z, camera.x, camera.z, fwd_x, fwd_z, cos_threshold):
                lp.draw(scene_shader, earthquake, elapsed_time)

        # Prédios e Casas (PBR red_brick / damaged_plaster com crossfade para broken_brick_wall / cracked_concrete_02)
        for b in all_buildings:
            b.draw(scene_shader, earthquake, elapsed_time)

        # Montanha (PBR rocky_terrain_02)
        scene_shader.set_uniform_int("u_mountain_stratum", 1)
        mountain.draw(scene_shader)
        scene_shader.set_uniform_int("u_mountain_stratum", 0)

        # Floresta / Árvores (tronco procedural + dry_river_pebbles sob raízes caídas)
        # — mesmo culling por ângulo de visão usado nos postes.
        for tree in forest:
            if _in_view(tree.x, tree.z, camera.x, camera.z, fwd_x, fwd_z, cos_threshold):
                tree.draw(scene_shader, earthquake, elapsed_time)

        scene_shader.stop()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Escombros de prédios/montanha, todos em 1-2 draw calls instanciados.
        debris_renderer.draw(view_matrix, proj_matrix, light_space_matrix, shadow_map.texture,
                              (camera.x, camera.y, camera.z))

        # 3. Partículas de fumaça e poeira com Billboards e Alpha Blending
        particle_system.draw(view_matrix, proj_matrix)

        # 4. HUD 2D com isolamento de profundidade
        hud.draw(
            WINDOW_SIZE[0], WINDOW_SIZE[1],
            earthquake=earthquake,
            camera=camera,
            buildings=buildings,
            houses=houses,
            fps=clock.get_fps(),
            lamp_posts=lamp_posts
        )

        pygame.display.flip()

    # -----------------------------------------------------------------------
    # Finalização e Limpeza de Memória GPU
    # -----------------------------------------------------------------------
    ground.cleanup()
    # Building não possui recursos GPU próprios (usa apenas meshes
    # compartilhados via shared.py) — nada a limpar aqui.
    mountain.cleanup()
    particle_system.cleanup()
    debris_renderer.cleanup()
    hud.cleanup()
    shadow_map.cleanup()
    sky.cleanup()
    scene_shader.cleanup()
    cleanup_textures()
    reset_shared_resources()
    reset_texture_cache()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
