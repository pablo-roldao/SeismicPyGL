import numpy as np
import pytest

from src.core.obj_loader import (
    parse_obj, compute_tangents,
    create_cube_mesh, create_quad_mesh, create_cylinder_mesh, create_plane_mesh,
)


def test_create_cube_mesh_shape():
    vdata, count, stride = create_cube_mesh()
    assert count == 36  # 6 faces * 2 tris * 3 verts
    assert stride == 44  # com tangente
    assert vdata.shape == (36 * 11,)


def test_create_cube_mesh_front_face_normal():
    vdata, count, stride = create_cube_mesh(width=2.0, height=2.0, depth=2.0)
    verts = vdata.reshape(-1, 11)
    # Primeiro vértice é da face frontal (Z+), normal (0,0,1).
    assert np.allclose(verts[0, 5:8], (0.0, 0.0, 1.0))
    assert np.allclose(verts[0, 0:3], (-1.0, 0.0, 1.0))


def test_create_quad_mesh_shape():
    vdata, count, stride = create_quad_mesh(size=2.0)
    assert count == 6
    assert stride == 32  # sem tangente
    verts = vdata.reshape(-1, 8)
    assert np.allclose(verts[:, 0:3].min(axis=0), (-1.0, -1.0, 0.0))
    assert np.allclose(verts[:, 0:3].max(axis=0), (1.0, 1.0, 0.0))


def test_create_cylinder_mesh_shape():
    vdata, count, stride = create_cylinder_mesh(slices=4)
    assert count == 4 * 6  # 4 fatias * 2 tris * 3 verts
    assert stride == 44


def test_create_cylinder_mesh_radius():
    vdata, count, stride = create_cylinder_mesh(base_radius=2.0, top_radius=0.5, height=3.0, slices=8)
    verts = vdata.reshape(-1, 11)
    base_points = verts[verts[:, 1] == 0.0][:, 0:3]
    top_points = verts[np.isclose(verts[:, 1], 3.0)][:, 0:3]
    base_radii = np.linalg.norm(base_points[:, [0, 2]], axis=1)
    top_radii = np.linalg.norm(top_points[:, [0, 2]], axis=1)
    assert np.allclose(base_radii, 2.0, atol=1e-4)
    assert np.allclose(top_radii, 0.5, atol=1e-4)


def test_create_plane_mesh_grid_shape():
    divisions = 4
    vdata, indices, count, stride = create_plane_mesh(size=8.0, divisions=divisions)
    assert count == (divisions + 1) ** 2
    assert len(indices) == divisions * divisions * 6
    assert stride == 44


def test_create_plane_mesh_corner_positions_match_cell_math():
    vdata, indices, count, stride = create_plane_mesh(size=8.0, divisions=4)
    verts = vdata.reshape(-1, 11)
    i00, i10, i11, i00b, i11b, i01 = indices[0:6]
    assert i00 == i00b and i11 == i11b

    step, half = 2.0, 4.0
    x0, z0 = -half, -half
    x1, z1 = x0 + step, z0 + step
    assert np.allclose(verts[i00, 0:3], (x0, 0.0, z0))
    assert np.allclose(verts[i10, 0:3], (x1, 0.0, z0))
    assert np.allclose(verts[i11, 0:3], (x1, 0.0, z1))
    assert np.allclose(verts[i01, 0:3], (x0, 0.0, z1))


def test_create_plane_mesh_uniform_tangent():
    vdata, indices, count, stride = create_plane_mesh(size=8.0, divisions=4)
    verts = vdata.reshape(-1, 11)
    assert np.allclose(verts[:, 8:11], (1.0, 0.0, 0.0))


def test_compute_tangents_known_gradient():
    # Triângulo no plano XZ com UV alinhado a X/Z, igual à grade do chão:
    # a tangente esperada (derivada da posição em U) é (1,0,0).
    verts = np.array([
        0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,  1.0, 0.0,  0.0, 1.0, 0.0,
        1.0, 0.0, 1.0,  1.0, 1.0,  0.0, 1.0, 0.0,
    ], dtype=np.float32)
    out, count, stride = compute_tangents(verts)
    tangents = out.reshape(-1, 11)[:, 8:11]
    assert np.allclose(tangents, (1.0, 0.0, 0.0), atol=1e-5)


def test_parse_obj_simple_triangle(tmp_path):
    obj_path = tmp_path / "triangle.obj"
    obj_path.write_text(
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "vt 0.0 0.0\n"
        "vt 1.0 0.0\n"
        "vt 0.0 1.0\n"
        "vn 0.0 0.0 1.0\n"
        "f 1/1/1 2/2/1 3/3/1\n"
    )
    vdata, count, stride = parse_obj(str(obj_path))
    assert count == 3
    assert stride == 44
    verts = vdata.reshape(-1, 11)
    assert np.allclose(verts[0, 0:3], (0.0, 0.0, 0.0))
    assert np.allclose(verts[1, 0:3], (1.0, 0.0, 0.0))
    assert np.allclose(verts[:, 5:8], (0.0, 0.0, 1.0))


def test_parse_obj_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_obj("/nonexistent/path/does_not_exist.obj")
