import pytest

from src.simulation.earthquake import EarthquakeSimulator


FAKE_PBR_SET = {"albedo": 0, "normal": 0, "roughness": 0}


@pytest.fixture(autouse=True)
def no_gpu_textures(monkeypatch):
    """
    Building.__init__ carrega texturas PBR via OpenGL (get_pbr_set -> glGenTextures),
    mesmo que update() em si seja pura física/dano — substitui por um dict fake para
    os testes rodarem sem contexto GL.
    """
    monkeypatch.setattr("src.world.building.get_pbr_set", lambda name: FAKE_PBR_SET)


def make_building(**kwargs):
    from src.world.building import Building
    defaults = dict(x=0.0, z=0.0, width=4.0, depth=4.0, height=10.0)
    defaults.update(kwargs)
    return Building(**defaults)


def make_strong_quake(epicenter=(0.0, 0.0)):
    eq = EarthquakeSimulator()
    eq.trigger(current_time=0.0, epicenter=epicenter, magnitude=8.5)
    return eq


def test_new_building_is_undamaged():
    b = make_building()
    assert b.damage == 0.0
    assert b.collapsing is False


def test_damage_accumulates_under_strong_shaking():
    b = make_building(x=0.0, z=0.0)
    b.resistance = 1000.0  # alto o bastante para nunca colapsar neste teste
    eq = make_strong_quake(epicenter=(0.0, 0.0))

    dt = 1 / 60.0
    t = 0.0
    for _ in range(30):
        t += dt
        b.update(eq, current_time=t, dt=dt, particle_system=None)

    assert b.damage > 0.0


def test_no_damage_without_shaking():
    b = make_building(x=0.0, z=0.0)
    eq = EarthquakeSimulator()  # nunca disparado (inactive)

    dt = 1 / 60.0
    for i in range(30):
        b.update(eq, current_time=i * dt, dt=dt, particle_system=None)

    assert b.damage == 0.0
    assert b.collapsing is False


def test_collapse_triggers_when_damage_exceeds_resistance():
    b = make_building(x=0.0, z=0.0)
    b.resistance = 0.05  # baixo o bastante para colapsar rápido e determinístico
    eq = make_strong_quake(epicenter=(0.0, 0.0))

    dt = 1 / 60.0
    t = 0.0
    for _ in range(120):
        t += dt
        b.update(eq, current_time=t, dt=dt, particle_system=None)
        if b.collapsing:
            break

    assert b.collapsing is True
    assert b.damage == b.resistance


def test_collapse_progress_advances_and_clamps():
    b = make_building(x=0.0, z=0.0)
    b.resistance = 0.05
    eq = make_strong_quake(epicenter=(0.0, 0.0))

    dt = 1 / 60.0
    t = 0.0
    # Passa do dano para o colapso (o próprio tick que dispara collapsing=True
    # ainda roda o ramo de dano, não o de collapse_progress).
    while not b.collapsing:
        t += dt
        b.update(eq, current_time=t, dt=dt, particle_system=None)
    t += dt
    b.update(eq, current_time=t, dt=dt, particle_system=None)

    progress_after_first_collapse_tick = b.collapse_progress
    assert progress_after_first_collapse_tick > 0.0

    # Roda tempo suficiente para ultrapassar COLLAPSE_DURATION várias vezes.
    for _ in range(int(b.COLLAPSE_DURATION / dt) * 3):
        t += dt
        b.update(eq, current_time=t, dt=dt, particle_system=None)

    assert b.collapse_progress == 1.0
