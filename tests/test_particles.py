"""
Testa a lógica vetorizada de ParticleSystem sem GPU: constrói via __new__
(pula __init__, que cria VBO/shader/textura) e monta manualmente os arrays
que __init__ normalmente prepararia. Mesma técnica usada informalmente no
benchmark de performance desta sessão (commit "perf(particles): ...").
"""
import numpy as np
import pytest

from src.simulation.particles import ParticleSystem


def make_headless_particle_system(max_particles=100):
    ps = ParticleSystem.__new__(ParticleSystem)
    ps.max_particles = max_particles
    ps._count = 0
    ps.pos = np.zeros((max_particles, 3), dtype=np.float32)
    ps.vel = np.zeros((max_particles, 3), dtype=np.float32)
    ps.size = np.zeros(max_particles, dtype=np.float32)
    ps.initial_size = np.zeros(max_particles, dtype=np.float32)
    ps.max_size = np.zeros(max_particles, dtype=np.float32)
    ps.alpha = np.zeros(max_particles, dtype=np.float32)
    ps.life = np.zeros(max_particles, dtype=np.float32)
    ps.lifetime = np.ones(max_particles, dtype=np.float32)
    ps.color = np.zeros((max_particles, 3), dtype=np.float32)
    ps._ambient_dust_budget = 0.0
    ps._leaf_budget = 0.0
    return ps


def test_emit_increases_count():
    ps = make_headless_particle_system()
    ps.emit((0.0, 1.0, 0.0), count=10)
    assert ps._count == 10


def test_emit_caps_at_max_particles():
    ps = make_headless_particle_system(max_particles=20)
    ps.emit((0.0, 1.0, 0.0), count=15)
    ps.emit((0.0, 1.0, 0.0), count=15)
    assert ps._count == 20


def test_emit_zero_count_is_noop():
    ps = make_headless_particle_system(max_particles=5)
    ps._count = 5
    ps.emit((0.0, 1.0, 0.0), count=10)
    assert ps._count == 5


def test_emitted_particles_start_at_full_alpha_and_zero_life():
    ps = make_headless_particle_system()
    ps.emit((1.0, 2.0, 3.0), count=5)
    assert np.all(ps.alpha[:5] == 1.0)
    assert np.all(ps.life[:5] == 0.0)


def test_update_advances_life_and_decays_alpha():
    ps = make_headless_particle_system()
    ps.emit((0.0, 1.0, 0.0), count=1)
    ps.lifetime[0] = 2.0
    alpha_before = ps.alpha[0]
    ps.update(dt=0.5)
    assert ps.life[0] == pytest.approx(0.5)
    assert ps.alpha[0] < alpha_before


def test_update_removes_expired_particles():
    ps = make_headless_particle_system()
    ps.emit((0.0, 1.0, 0.0), count=3)
    ps.lifetime[:3] = 1.0
    ps.update(dt=2.0)  # ultrapassa o lifetime de todas
    assert ps._count == 0


def test_update_compacts_surviving_particles_to_front():
    ps = make_headless_particle_system()
    ps.emit((0.0, 1.0, 0.0), count=2)
    ps.lifetime[0] = 0.1  # expira logo
    ps.lifetime[1] = 10.0  # sobrevive
    ps.pos[1] = (7.0, 8.0, 9.0)
    ps.vel[1] = (0.0, 0.0, 0.0)  # posição estável para a asserção abaixo
    ps.update(dt=0.5)
    assert ps._count == 1
    assert np.allclose(ps.pos[0], (7.0, 8.0, 9.0))


def test_update_noop_when_no_particles():
    ps = make_headless_particle_system()
    ps.update(dt=1 / 60.0)  # não deve levantar exceção
    assert ps._count == 0


def test_emit_ambient_adds_dust_over_time():
    ps = make_headless_particle_system()
    forest = []
    for _ in range(120):  # tempo suficiente para o budget de poeira acumular
        ps.emit_ambient((0.0, 5.0, 0.0), forest, dt=1 / 60.0)
    assert ps._count > 0


def test_emit_ambient_respects_max_particles():
    ps = make_headless_particle_system(max_particles=3)
    forest = []
    for _ in range(600):
        ps.emit_ambient((0.0, 5.0, 0.0), forest, dt=1 / 60.0)
    assert ps._count <= 3
