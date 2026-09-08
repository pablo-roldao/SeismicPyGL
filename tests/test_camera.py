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
