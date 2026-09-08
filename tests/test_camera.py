import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.camera import FreeCamera


def test_zoom_clamps_to_bounds():
    cam = FreeCamera(fov=55.0, fov_min=20.0, fov_max=90.0, zoom_speed=3.0)
    cam.zoom(scroll_amount=100.0)  # zoom in bem além do limite
    assert cam.fov == 20.0
    cam.zoom(scroll_amount=-1000.0)  # zoom out bem além do limite
    assert cam.fov == 90.0


def test_process_mouse_moving_right_turns_camera_right():
    """Regressão: eixo X do mouse estava invertido (yaw -= dx em vez de += dx)."""
    cam = FreeCamera()
    fwd_before = cam._base_forward_vector()
    cam.process_mouse(rel_x=50, rel_y=0)
    fwd_after = cam._base_forward_vector()
    assert fwd_after[0] > fwd_before[0]


def test_process_mouse_moving_down_looks_down():
    cam = FreeCamera()
    pitch_before = cam.pitch
    cam.process_mouse(rel_x=0, rel_y=50)
    assert cam.pitch < pitch_before


def test_process_mouse_pitch_clamped():
    cam = FreeCamera()
    for _ in range(50):
        cam.process_mouse(rel_x=0, rel_y=50)
    assert cam.pitch >= -84.0
    for _ in range(50):
        cam.process_mouse(rel_x=0, rel_y=-50)
    assert cam.pitch <= 84.0


def test_process_mouse_ignores_anomalous_deltas():
    cam = FreeCamera()
    yaw_before, pitch_before = cam.yaw, cam.pitch
    cam.process_mouse(rel_x=300, rel_y=300)
    assert cam.yaw == yaw_before
    assert cam.pitch == pitch_before


def test_add_trauma_clamps_to_unit_range():
    cam = FreeCamera()
    cam.add_trauma(2.0)
    assert cam.trauma == 1.0
    cam.add_trauma(-5.0)
    assert cam.trauma == 0.0


def test_update_trauma_decays_over_time():
    cam = FreeCamera()
    cam.add_trauma(1.0)
    cam.update_trauma(dt=0.5)
    assert 0.0 < cam.trauma < 1.0


def test_update_trauma_zero_shake_when_no_trauma():
    cam = FreeCamera()
    cam.update_trauma(dt=0.1)
    assert cam._shake_yaw == 0.0
    assert cam._shake_pitch == 0.0
    assert cam._shake_roll == 0.0
    assert cam._shake_pos == (0.0, 0.0, 0.0)


def test_reset_view_restores_initial_pose():
    cam = FreeCamera()
    cam.x, cam.y, cam.z = 10.0, 20.0, 30.0
    cam.yaw, cam.pitch = 45.0, 10.0
    cam.add_trauma(1.0)
    cam.reset_view()
    assert (cam.x, cam.y, cam.z, cam.yaw, cam.pitch) == cam._initial_pose
    assert cam.trauma == 0.0


def test_base_forward_vector_at_yaw_zero_points_plus_x():
    cam = FreeCamera(yaw=0.0, pitch=0.0)
    fx, fy, fz = cam._base_forward_vector()
    assert np.isclose(fx, 1.0, atol=1e-6)
    assert np.isclose(fy, 0.0, atol=1e-6)
    assert np.isclose(fz, 0.0, atol=1e-6)


def test_forward_vector_matches_base_forward_vector():
    cam = FreeCamera(yaw=33.0, pitch=-12.0)
    assert cam.forward_vector() == cam._base_forward_vector()


def test_get_view_matrix_identity_case():
    cam = FreeCamera(position=(0.0, 5.0, 1.0), yaw=-90.0, pitch=0.0)
    view = cam.get_view_matrix()
    assert view.shape == (4, 4)
    # Câmera olhando para -Z a partir de (0,5,1): eye e target diferem só em Z.
    assert np.allclose(view[0:3, 0:3], np.eye(3), atol=1e-6)


# --- Colisão com objetos do mundo -----------------------------------------

def test_push_out_of_rect_stops_camera_near_wall():
    cam = FreeCamera()
    # Prédio 4x4 centrado na origem (half_w=half_d=2.0); câmera logo fora da borda.
    nx, nz = cam._push_out_of_rect(0.0, 2.1, 0.0, 0.0, 2.0, 2.0)
    assert nz >= 2.0 + cam.collision_radius - 1e-6


def test_push_out_of_rect_unaffected_when_far_away():
    cam = FreeCamera()
    nx, nz = cam._push_out_of_rect(0.0, 10.0, 0.0, 0.0, 2.0, 2.0)
    assert (nx, nz) == (0.0, 10.0)


def test_push_out_of_rect_resolves_tunneling_from_inside():
    """Se a câmera pular para dentro do retângulo num único passo, deve ser
    empurrada para a face mais próxima, não ficar presa lá dentro."""
    cam = FreeCamera()
    nx, nz = cam._push_out_of_rect(0.0, 1.9, 0.0, 0.0, 2.0, 2.0)
    outside = (
        nz >= 2.0 + cam.collision_radius - 1e-6 or nz <= -(2.0 + cam.collision_radius) + 1e-6
        or abs(nx) >= 2.0 + cam.collision_radius - 1e-6
    )
    assert outside


def test_push_out_of_circle_stops_camera_near_tree():
    cam = FreeCamera()
    nx, nz = cam._push_out_of_circle(5.0, 5.1, 5.0, 5.0, 0.3)
    dist = ((nx - 5.0) ** 2 + (nz - 5.0) ** 2) ** 0.5
    assert dist >= 0.3 + cam.collision_radius - 1e-6


def _walk_forward(cam, dt, steps, **obstacle_kwargs):
    import pygame

    class HeldW(dict):
        def __getitem__(self, key):
            return key == pygame.K_w

    original_get_pressed = pygame.key.get_pressed
    pygame.key.get_pressed = lambda: HeldW()
    try:
        for _ in range(steps):
            cam.process_keyboard(dt, **obstacle_kwargs)
    finally:
        pygame.key.get_pressed = original_get_pressed


def test_process_keyboard_stops_at_building_without_tunneling():
    cam = FreeCamera(position=(0.0, 5.0, 10.0), yaw=-90.0, pitch=0.0)  # olha para -Z
    building = SimpleNamespace(x=0.0, z=0.0, width=4.0, depth=4.0)
    _walk_forward(cam, dt=1 / 60.0, steps=600, buildings=[building])
    assert cam.z >= 2.0 + cam.collision_radius - 1e-3


def _walk_forward_never_entering(cam, dt, steps, obstacle_x, obstacle_z, obstacle_radius, **obstacle_kwargs):
    """
    Anda para frente e confere, a cada frame, que a câmera nunca fica mais
    perto do centro do obstáculo circular do que obstacle_radius +
    collision_radius. Contra um obstáculo redondo e fino, empurrar para
    frente sem parar faz a câmera "deslizar" pela lateral e ultrapassá-lo
    (como contornar um poste na vida real) — o que importa é que ela nunca
    entra no raio do obstáculo, não que fique bloqueada para sempre.
    """
    import pygame

    class HeldW(dict):
        def __getitem__(self, key):
            return key == pygame.K_w

    original_get_pressed = pygame.key.get_pressed
    pygame.key.get_pressed = lambda: HeldW()
    min_allowed = obstacle_radius + cam.collision_radius
    try:
        for _ in range(steps):
            cam.process_keyboard(dt, **obstacle_kwargs)
            dist = math.hypot(cam.x - obstacle_x, cam.z - obstacle_z)
            assert dist >= min_allowed - 1e-6, f"câmera entrou no obstáculo: dist={dist}, min={min_allowed}"
    finally:
        pygame.key.get_pressed = original_get_pressed


def test_process_keyboard_standing_tree_never_entered():
    cam = FreeCamera(position=(0.0, 5.0, 10.0), yaw=-90.0, pitch=0.0)
    tree = SimpleNamespace(x=0.0, z=0.0, trunk_radius=0.3, falling=False)
    _walk_forward_never_entering(cam, dt=1 / 60.0, steps=600, obstacle_x=0.0, obstacle_z=0.0,
                                  obstacle_radius=0.3, trees=[tree])


def test_process_keyboard_fallen_tree_does_not_block():
    cam = FreeCamera(position=(0.0, 5.0, 10.0), yaw=-90.0, pitch=0.0)
    tree = SimpleNamespace(x=0.0, z=0.0, trunk_radius=0.3, falling=True)
    _walk_forward(cam, dt=1 / 60.0, steps=600, trees=[tree])
    assert cam.z < 0.3  # atravessou, já que a árvore caída não bloqueia


def test_process_keyboard_mountain_never_entered():
    cam = FreeCamera(position=(0.0, 5.0, 10.0), yaw=-90.0, pitch=0.0)
    mountain = SimpleNamespace(x=0.0, z=0.0, base_radius=3.0)
    _walk_forward_never_entering(cam, dt=1 / 60.0, steps=600, obstacle_x=0.0, obstacle_z=0.0,
                                  obstacle_radius=3.0, mountain=mountain)


def test_process_keyboard_standing_light_pole_never_entered():
    cam = FreeCamera(position=(0.0, 5.0, 10.0), yaw=-90.0, pitch=0.0)
    pole = SimpleNamespace(x=0.0, z=0.0, falling=False)
    _walk_forward_never_entering(cam, dt=1 / 60.0, steps=600, obstacle_x=0.0, obstacle_z=0.0,
                                  obstacle_radius=cam.LIGHT_POLE_COLLISION_RADIUS, light_poles=[pole])


def test_process_keyboard_fallen_light_pole_does_not_block():
    cam = FreeCamera(position=(0.0, 5.0, 10.0), yaw=-90.0, pitch=0.0)
    pole = SimpleNamespace(x=0.0, z=0.0, falling=True)
    _walk_forward(cam, dt=1 / 60.0, steps=600, light_poles=[pole])
    assert cam.z < 0.0  # atravessou


def test_process_keyboard_without_obstacles_still_moves_normally():
    """Regressão: parâmetros de colisão são opcionais e não quebram o uso sem obstáculos."""
    cam = FreeCamera(position=(0.0, 5.0, 10.0), yaw=-90.0, pitch=0.0)
    _walk_forward(cam, dt=1 / 60.0, steps=10)
    assert cam.z < 10.0
