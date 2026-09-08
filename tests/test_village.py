import pytest

FAKE_PBR_SET = {"albedo": 0, "normal": 0, "roughness": 0}


@pytest.fixture(autouse=True)
def no_gpu_textures(monkeypatch):
    """generate_village constrói Building/Street/LightPole, que carregam PBR via GL."""
    monkeypatch.setattr("src.world.building.get_pbr_set", lambda name: FAKE_PBR_SET)
    monkeypatch.setattr("src.world.street.get_pbr_set", lambda name: FAKE_PBR_SET)
    monkeypatch.setattr("src.world.light_pole.get_pbr_set", lambda name: FAKE_PBR_SET)


def test_generate_village_respects_building_and_house_count():
    from src.world.village import generate_village
    buildings, houses, streets, light_poles = generate_village(
        center=(0.0, 0.0), building_count=4, house_count=8
    )
    assert len(buildings) == 4
    assert len(houses) == 8


def test_generate_village_produces_streets_and_light_poles():
    from src.world.village import generate_village
    buildings, houses, streets, light_poles = generate_village(
        center=(0.0, 0.0), building_count=4, house_count=8
    )
    assert len(streets) > 0
    assert len(light_poles) > 0
