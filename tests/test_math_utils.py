import math

import numpy as np

from src.core.math_utils import (
    identity, perspective, look_at, ortho,
    translate, scale, rotate_x, rotate_y, rotate_z,
    to_gl_matrix, perlin1d, perlin2d,
)


def apply(matrix, point_xyz):
    """Aplica a matriz 4x4 a um ponto (x, y, z), retornando (x, y, z)."""
    v = np.array([*point_xyz, 1.0], dtype=np.float32)
    return tuple((matrix @ v)[:3])


def test_identity():
    assert np.allclose(identity(), np.eye(4))


def test_translate_moves_point():
    m = translate(1.0, 2.0, 3.0)
    assert np.allclose(apply(m, (0.0, 0.0, 0.0)), (1.0, 2.0, 3.0))


def test_scale_scales_point():
    m = scale(2.0, 3.0, 4.0)
    assert np.allclose(apply(m, (1.0, 1.0, 1.0)), (2.0, 3.0, 4.0))


def test_rotate_z_90_degrees():
    m = rotate_z(90.0)
    x, y, z = apply(m, (1.0, 0.0, 0.0))
    assert math.isclose(x, 0.0, abs_tol=1e-6)
    assert math.isclose(y, 1.0, abs_tol=1e-6)
    assert math.isclose(z, 0.0, abs_tol=1e-6)


def test_rotate_x_90_degrees():
    m = rotate_x(90.0)
    x, y, z = apply(m, (0.0, 1.0, 0.0))
    assert math.isclose(x, 0.0, abs_tol=1e-6)
    assert math.isclose(y, 0.0, abs_tol=1e-6)
    assert math.isclose(z, 1.0, abs_tol=1e-6)


def test_rotate_y_90_degrees():
    m = rotate_y(90.0)
    x, y, z = apply(m, (0.0, 0.0, 1.0))
    assert math.isclose(x, 1.0, abs_tol=1e-6)
    assert math.isclose(y, 0.0, abs_tol=1e-6)
    assert math.isclose(z, 0.0, abs_tol=1e-6)


def test_perspective_known_values():
    m = perspective(fov_deg=90.0, aspect=1.0, near=0.1, far=100.0)
    assert math.isclose(m[0, 0], 1.0, abs_tol=1e-6)
    assert math.isclose(m[1, 1], 1.0, abs_tol=1e-6)
    assert math.isclose(m[3, 2], -1.0, abs_tol=1e-6)
    assert m[2, 2] < 0.0


def test_perspective_aspect_ratio_scales_x():
    m = perspective(fov_deg=90.0, aspect=2.0, near=0.1, far=100.0)
    assert math.isclose(m[0, 0], 0.5, abs_tol=1e-6)


def test_ortho_symmetric_bounds():
    m = ortho(left=-10.0, right=10.0, bottom=-5.0, top=5.0, near=-1.0, far=1.0)
    assert math.isclose(m[0, 0], 0.1, abs_tol=1e-6)
    assert math.isclose(m[1, 1], 0.2, abs_tol=1e-6)
    assert math.isclose(m[0, 3], 0.0, abs_tol=1e-6)
    assert math.isclose(m[1, 3], 0.0, abs_tol=1e-6)


def test_look_at_identity_case():
    """Câmera na origem olhando para -Z com up=+Y reproduz a identidade (convenção OpenGL)."""
    m = look_at(eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0))
    assert np.allclose(m, np.eye(4), atol=1e-6)


def test_to_gl_matrix_is_transpose():
    m = translate(1.0, 2.0, 3.0)
    gl_m = to_gl_matrix(m)
    assert np.allclose(gl_m, m.T)
    assert gl_m.flags["C_CONTIGUOUS"]


def test_perlin1d_zero_at_integer_lattice():
    """Propriedade clássica do ruído de Perlin: é exatamente 0 em coordenadas inteiras."""
    assert perlin1d(0.0) == 0.0
    assert perlin1d(1.0) == 0.0
    assert perlin1d(5.0) == 0.0


def test_perlin2d_zero_at_integer_lattice():
    assert perlin2d(0.0, 0.0) == 0.0
    assert perlin2d(2.0, 3.0) == 0.0


def test_perlin1d_deterministic_for_same_seed():
    a = perlin1d(1.234, seed=7)
    b = perlin1d(1.234, seed=7)
    assert a == b


def test_perlin1d_differs_across_seeds():
    a = perlin1d(1.234, seed=1)
    b = perlin1d(1.234, seed=2)
    assert a != b
