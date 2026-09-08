"""
Testa só a parte pura de mountain.py (RockDebris). A classe Mountain em si
constrói uma malha procedural própria (Mesh(...)) já no __init__, então
precisa de contexto GL de verdade — coberta em test_rendering_smoke.py.
"""
import numpy as np

from src.world.mountain import RockDebris


def test_new_rock_is_alive():
    rock = RockDebris(x=0.0, y=5.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, lifetime=6.0)
    assert rock.alive is True


def test_rock_expires_after_lifetime():
    rock = RockDebris(x=0.0, y=5.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, lifetime=1.0)
    rock.update(dt=1.5)
    assert rock.alive is False


def test_rock_falls_under_gravity():
    rock = RockDebris(x=0.0, y=5.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, size=0.1)
    y_before = rock.y
    rock.update(dt=0.1)
    assert rock.y < y_before
    assert rock.vy < 0.0


def test_rock_bounces_at_ground():
    rock = RockDebris(x=0.0, y=0.1, z=0.0, vx=0.0, vy=-5.0, vz=0.0, size=0.4)
    rock.update(dt=0.1)
    assert rock.y == rock.size * 0.5
    assert rock.vy > 0.0  # quicou (velocidade invertida e amortecida)


def test_rock_instance_data_matrix_places_rock_at_position():
    rock = RockDebris(x=1.0, y=2.0, z=3.0, vx=0.0, vy=0.0, vz=0.0, size=0.5,
                       color=(0.5, 0.5, 0.5, 1.0))
    model, color = rock.instance_data()
    translation = model[0:3, 3]
    assert np.allclose(translation, (1.0, 2.0, 3.0))
    assert color == (0.5, 0.5, 0.5, 1.0)
