"""
Testes de "fumaça" com um contexto OpenGL real (fixture gl_context em
conftest.py): constroem as classes de renderização, chamam update()/draw()/
cleanup() com argumentos plausíveis e conferem glGetError() == GL_NO_ERROR.
Não fazem asserção visual/pixel a pixel — só que as chamadas GL não quebram.
"""
import numpy as np
import pytest
from OpenGL.GL import glGetError, GL_NO_ERROR

from src.core.mesh import Mesh
from src.core.shader import ShaderProgram
from src.simulation.earthquake import EarthquakeSimulator

pytestmark = pytest.mark.usefixtures("gl_context")

IDENTITY = np.eye(4, dtype=np.float32)


def assert_no_gl_error(label=""):
    err = glGetError()
    assert err == GL_NO_ERROR, f"GL error {label}: {err}"


def make_scene_shader():
    shader = ShaderProgram.from_files("assets/shaders/scene.vert", "assets/shaders/scene.frag")
    shader.use()
    return shader


# --- Mesh ---------------------------------------------------------------

def test_mesh_create_cube_draw_cleanup():
    mesh = Mesh.create_cube()
    assert_no_gl_error("create_cube")
    mesh.draw()
    assert_no_gl_error("cube draw")
    mesh.cleanup()
    assert_no_gl_error("cube cleanup")


def test_mesh_create_plane_indexed_draw_cleanup():
    mesh = Mesh.create_plane(size=10.0, divisions=4)
    assert mesh.ebo is not None
    mesh.draw()
    assert_no_gl_error("plane draw")
    mesh.cleanup()
    assert_no_gl_error("plane cleanup")


def test_mesh_create_cylinder_draw_cleanup():
    mesh = Mesh.create_cylinder(slices=8)
    mesh.draw()
    assert_no_gl_error("cylinder draw")
    mesh.cleanup()


def test_mesh_create_quad_draw_cleanup():
    mesh = Mesh.create_quad()
    mesh.draw()
    assert_no_gl_error("quad draw")
    mesh.cleanup()


def test_mesh_from_obj_draw_cleanup(tmp_path):
    obj_path = tmp_path / "tri.obj"
    obj_path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 0 1\n"
        "vn 0 0 1\n"
        "f 1/1/1 2/2/1 3/3/1\n"
    )
    mesh = Mesh.from_obj(str(obj_path))
    mesh.draw()
    assert_no_gl_error("from_obj draw")
    mesh.cleanup()


# --- ShaderProgram --------------------------------------------------------

def test_shader_program_from_files_and_uniforms():
    shader = ShaderProgram.from_files("assets/shaders/billboard.vert", "assets/shaders/billboard.frag")
    shader.use()
    shader.set_uniform_mat4("u_view", IDENTITY)
    shader.set_uniform_mat4("u_projection", IDENTITY)
    shader.set_uniform_vec3("u_camera_right", (1.0, 0.0, 0.0))
    shader.set_uniform_vec3("u_camera_up", (0.0, 1.0, 0.0))
    shader.set_uniform_int("u_texture", 0)
    shader.stop()
    assert_no_gl_error("billboard shader uniforms")
    shader.cleanup()
    assert_no_gl_error("shader cleanup")


# --- Ground ---------------------------------------------------------------

def test_ground_draw_cleanup():
    from src.world.ground import Ground
    ground = Ground(size=20.0, divisions=4)
    assert_no_gl_error("ground construct")
    eq = EarthquakeSimulator()
    ground.draw(eq, 0.0, IDENTITY, IDENTITY)
    assert_no_gl_error("ground draw")
    ground.cleanup()
    assert_no_gl_error("ground cleanup")


# --- Street -----------------------------------------------------------

def test_street_draw():
    from src.world.street import Street
    shader = make_scene_shader()
    street = Street(x=0.0, z=0.0, width=2.5, length=20.0)
    street.draw(shader)
    assert_no_gl_error("street draw")
    shader.stop()
    shader.cleanup()


# --- Building (intacto e colapsando) ------------------------------------

def test_building_draw_intact_and_collapsing():
    from src.world.building import Building
    shader = make_scene_shader()
    eq = EarthquakeSimulator()

    b = Building(x=0.0, z=0.0, width=2.0, depth=2.0, height=6.0)
    b.draw(shader, eq, 0.0)
    assert_no_gl_error("intact building draw")

    b.collapsing = True
    b.collapse_progress = 0.5
    b._spawn_debris()
    b.draw(shader, eq, 0.0)
    assert_no_gl_error("collapsing building draw")

    shader.stop()
    shader.cleanup()


# --- Mountain -----------------------------------------------------------

def test_mountain_construct_spawn_rock_draw_cleanup():
    from src.world.mountain import Mountain
    shader = make_scene_shader()

    mountain = Mountain(x=0.0, z=0.0, base_radius=4.0, height=6.0, bands=3, slices=6)
    assert_no_gl_error("mountain construct")
    mountain._spawn_rock()
    mountain.draw(shader)
    assert_no_gl_error("mountain draw")

    shader.stop()
    shader.cleanup()
    mountain.cleanup()
    assert_no_gl_error("mountain cleanup")


# --- Tree -----------------------------------------------------------------

def test_tree_draw():
    from src.world.nature import Tree
    shader = make_scene_shader()
    eq = EarthquakeSimulator()

    tree = Tree(x=0.0, z=0.0)
    tree.draw(shader, eq, 0.0)
    assert_no_gl_error("tree draw")

    shader.stop()
    shader.cleanup()


# --- LightPole --------------------------------------------------------

def test_light_pole_draw():
    from src.world.light_pole import LightPole
    shader = make_scene_shader()
    eq = EarthquakeSimulator()

    pole = LightPole(x=0.0, z=0.0)
    pole.draw(shader, eq, 0.0)
    assert_no_gl_error("light pole draw")

    shader.stop()
    shader.cleanup()


# --- DebrisRenderer -----------------------------------------------------

def test_debris_renderer_collect_and_draw():
    from src.world.debris_renderer import DebrisRenderer
    from src.world.building import Building
    from src.world.mountain import Mountain

    renderer = DebrisRenderer(max_instances=64)
    assert_no_gl_error("debris renderer construct")

    b = Building(x=0.0, z=0.0, width=2.0, depth=2.0, height=6.0)
    b._spawn_debris()
    for d in b.debris:
        d.released = True

    mountain = Mountain(x=10.0, z=10.0, base_radius=3.0, height=4.0, bands=2, slices=6)
    mountain._spawn_rock()

    renderer.collect([b])
    renderer.collect_rocks(mountain)
    renderer.draw(IDENTITY, IDENTITY, IDENTITY, 0, (0.0, 5.0, 0.0))
    assert_no_gl_error("debris renderer draw")

    renderer.cleanup()
    mountain.cleanup()
    assert_no_gl_error("debris renderer cleanup")


# --- ParticleSystem (construção real, com GPU) ---------------------------

def test_particle_system_real_construct_emit_update_draw_cleanup():
    from src.simulation.particles import ParticleSystem

    ps = ParticleSystem(max_particles=32)
    assert_no_gl_error("particle system construct")
    ps.emit((0.0, 1.0, 0.0), count=10)
    ps.update(dt=1 / 60.0)
    ps.draw(IDENTITY, IDENTITY)
    assert_no_gl_error("particle system draw")
    ps.cleanup()
    assert_no_gl_error("particle system cleanup")


# --- ShadowMap --------------------------------------------------------

def test_shadow_map_begin_end_cleanup():
    from src.rendering.shadow_map import ShadowMap
    shadow_map = ShadowMap(size=64)
    assert_no_gl_error("shadow map construct")
    shadow_map.begin(IDENTITY)
    shadow_map.end(320, 240)
    assert_no_gl_error("shadow map begin/end")
    shadow_map.cleanup()
    assert_no_gl_error("shadow map cleanup")


# --- Sky --------------------------------------------------------------

def test_sky_draw_cleanup():
    from src.rendering.sky import Sky
    sky = Sky()
    assert_no_gl_error("sky construct")
    sky.draw(IDENTITY, IDENTITY, elapsed_time=0.0)
    assert_no_gl_error("sky draw")
    sky.cleanup()
    assert_no_gl_error("sky cleanup")


# --- HUD ----------------------------------------------------------------

def test_hud_draw_cleanup():
    from src.rendering.hud import HUD
    from src.world.building import Building

    hud = HUD(width=320, height=240)
    assert_no_gl_error("hud construct")
    eq = EarthquakeSimulator()

    class FakeCamera:
        x = y = z = 0.0
        trauma = 0.0

    b = Building(x=0.0, z=0.0, width=2.0, depth=2.0, height=6.0)
    hud.draw(320, 240, eq, FakeCamera(), [b], [], fps=60.0, lamp_posts=[])
    assert_no_gl_error("hud draw")
    hud.cleanup()
    assert_no_gl_error("hud cleanup")


# --- Camera.apply() (único método de FreeCamera que precisa de GL/GLU) ----

def test_camera_apply():
    from src.core.camera import FreeCamera
    cam = FreeCamera()
    cam.apply()
    assert_no_gl_error("camera apply")


# --- shared.py: cache de PBR/malhas e reset (roda por último) -----------

def test_shared_pbr_cache_returns_same_object():
    from src.world.shared import get_pbr_set
    a = get_pbr_set("clean_asphalt")
    b = get_pbr_set("clean_asphalt")
    assert a is b


def test_shared_mesh_getters_cache_and_reset():
    from src.world import shared as shared_module
    from src.core.texture import reset_texture_cache

    mesh1 = shared_module.get_shared_cube_mesh()
    mesh2 = shared_module.get_shared_cube_mesh()
    assert mesh1 is mesh2

    shared_module.reset_shared_resources()
    reset_texture_cache()
    assert_no_gl_error("after reset_shared_resources")

    mesh3 = shared_module.get_shared_cube_mesh()
    assert mesh3 is not mesh1
    mesh3.draw()
    assert_no_gl_error("re-created shared mesh draw")
