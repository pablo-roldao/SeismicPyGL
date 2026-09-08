import pytest

from src.simulation.earthquake import EarthquakeSimulator

FAKE_PBR_SET = {"albedo": 0, "normal": 0, "roughness": 0}


class FakeParticleSystem:
    def __init__(self):
        self.emitted = []

    def emit(self, position, count=35, spread=1.2, base_speed=1.8):
        self.emitted.append((position, count, spread, base_speed))


@pytest.fixture(autouse=True)
def no_gpu_textures(monkeypatch):
    monkeypatch.setattr("src.world.light_pole.get_pbr_set", lambda name: FAKE_PBR_SET)


def make_pole(**kwargs):
    from src.world.light_pole import LightPole
    defaults = dict(x=0.0, z=0.0)
    defaults.update(kwargs)
    return LightPole(**defaults)


def test_new_pole_is_standing():
    pole = make_pole()
    assert pole.falling is False
    assert pole.fallen is False
    assert pole.lamp_lit is True


def test_reset_restores_initial_state():
    pole = make_pole(x=1.0, z=2.0)
    pole.x, pole.z = 99.0, 99.0
    pole.falling = True
    pole.fall_progress = 0.5
    pole.fallen = True
    pole.lamp_lit = False
    pole.reset()
    assert (pole.x, pole.z) == (1.0, 2.0)
    assert pole.falling is False
    assert pole.fall_progress == 0.0
    assert pole.fallen is False
    assert pole.lamp_lit is True


def test_fall_progress_advances_and_clamps():
    pole = make_pole()
    pole.falling = True
    eq = EarthquakeSimulator()  # inativo
    dt = 1 / 60.0
    for _ in range(int(1.0 / pole.fall_speed / dt) + 20):
        pole.update(eq, current_time=0.0, dt=dt, particle_system=None)
    assert pole.fall_progress == 1.0
    assert pole.fallen is True
    assert pole.lamp_lit is False


def test_impact_emits_particles_once():
    pole = make_pole()
    pole.falling = True
    pole.fall_progress = 0.999
    eq = EarthquakeSimulator()
    fake_particles = FakeParticleSystem()
    dt = 1 / 60.0

    pole.update(eq, current_time=0.0, dt=dt, particle_system=fake_particles)
    assert pole.fallen is True
    assert len(fake_particles.emitted) == 1

    # Não deve emitir de novo em atualizações seguintes.
    pole.update(eq, current_time=0.0, dt=dt, particle_system=fake_particles)
    assert len(fake_particles.emitted) == 1


def test_falling_triggers_under_strong_shake_with_forced_roll(monkeypatch):
    pole = make_pole(x=0.0, z=0.0)
    eq = EarthquakeSimulator()
    eq.trigger(current_time=0.0, epicenter=(0.0, 0.0), magnitude=8.5)
    monkeypatch.setattr("src.world.light_pole.random.random", lambda: 0.0)

    dt = 1 / 60.0
    pole.update(eq, current_time=0.05, dt=dt, particle_system=None)
    assert pole.falling is True


def test_no_fall_without_shake():
    pole = make_pole()
    eq = EarthquakeSimulator()  # nunca disparado
    dt = 1 / 60.0
    for i in range(60):
        pole.update(eq, current_time=i * dt, dt=dt, particle_system=None)
    assert pole.falling is False
