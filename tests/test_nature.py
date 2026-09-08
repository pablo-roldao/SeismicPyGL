from types import SimpleNamespace

import pytest

from src.simulation.earthquake import EarthquakeSimulator

FAKE_PBR_SET = {"albedo": 0, "normal": 0, "roughness": 0}


@pytest.fixture(autouse=True)
def no_gpu_textures(monkeypatch):
    monkeypatch.setattr("src.world.nature.get_pbr_set", lambda name: FAKE_PBR_SET)


def make_tree(**kwargs):
    from src.world.nature import Tree
    defaults = dict(x=0.0, z=0.0)
    defaults.update(kwargs)
    return Tree(**defaults)


def test_new_tree_is_standing():
    tree = make_tree()
    assert tree.falling is False
    assert tree.fall_progress == 0.0


def test_reset_clears_falling_state():
    tree = make_tree()
    tree.falling = True
    tree.fall_progress = 0.7
    tree.reset()
    assert tree.falling is False
    assert tree.fall_progress == 0.0


def test_fall_progress_advances_and_clamps_once_falling():
    tree = make_tree()
    tree.falling = True
    eq = EarthquakeSimulator()  # inativo: update() só mexe em fall_progress
    dt = 1 / 60.0
    for _ in range(int(1.5 / tree.fall_speed / dt) + 10):
        tree.update(eq, current_time=0.0, dt=dt)
    assert tree.fall_progress == 1.0


def test_falling_triggers_under_strong_shake_with_forced_roll(monkeypatch):
    tree = make_tree(x=0.0, z=0.0)
    eq = EarthquakeSimulator()
    eq.trigger(current_time=0.0, epicenter=(0.0, 0.0), magnitude=8.5)
    monkeypatch.setattr("src.world.nature.random.random", lambda: 0.0)

    dt = 1 / 60.0
    tree.update(eq, current_time=0.05, dt=dt)
    assert tree.falling is True


def test_no_fall_without_shake():
    tree = make_tree()
    eq = EarthquakeSimulator()  # nunca disparado
    dt = 1 / 60.0
    for i in range(60):
        tree.update(eq, current_time=i * dt, dt=dt)
    assert tree.falling is False


def test_generate_forest_respects_count():
    from src.world.nature import generate_forest
    forest = generate_forest(count=10, center=(0.0, 0.0), radius_range=(5.0, 10.0))
    assert len(forest) == 10


def test_generate_forest_avoids_zones():
    from src.world.nature import generate_forest
    forest = generate_forest(
        count=15, center=(0.0, 0.0), radius_range=(0.0, 20.0),
        avoid_zones=[(0.0, 0.0, 15.0)],
    )
    for tree in forest:
        assert (tree.x ** 2 + tree.z ** 2) ** 0.5 >= 15.0


def test_generate_forest_avoids_buildings():
    from src.world.nature import generate_forest
    fake_building = SimpleNamespace(x=0.0, z=0.0, width=2.0, depth=2.0)
    forest = generate_forest(
        count=10, center=(0.0, 0.0), radius_range=(0.0, 20.0),
        avoid_buildings=[fake_building],
    )
    for tree in forest:
        hw, hd = 1.0 + 1.2, 1.0 + 1.2
        assert not (abs(tree.x - 0.0) < hw and abs(tree.z - 0.0) < hd)
