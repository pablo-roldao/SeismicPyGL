"""
Testa a lógica de dispatch por nome de arquivo em load_texture() (fallback
procedural quando o arquivo não existe em disco), sem tocar GL: substitui
as funções de criação de textura por gravadores fake.
"""
import pytest

from src.core import texture as texture_module


@pytest.fixture(autouse=True)
def record_texture_creators(monkeypatch):
    calls = []
    for name in (
        "create_smoke_particle_texture", "create_grass_texture",
        "create_asphalt_texture", "get_flat_normal_texture",
        "get_default_roughness_texture", "create_concrete_texture",
    ):
        def make_recorder(fn_name):
            def recorder(*args, **kwargs):
                calls.append(fn_name)
                return 0
            return recorder
        monkeypatch.setattr(texture_module, name, make_recorder(name))
    return calls


def test_missing_smoke_file_uses_smoke_texture(record_texture_creators):
    texture_module.load_texture("/fake/dir/smoke_particle.png")
    assert record_texture_creators == ["create_smoke_particle_texture"]


def test_missing_grass_file_uses_grass_texture(record_texture_creators):
    texture_module.load_texture("/fake/dir/sparse_grass_diff.jpg")
    assert record_texture_creators == ["create_grass_texture"]


def test_missing_asphalt_file_uses_asphalt_texture(record_texture_creators):
    texture_module.load_texture("/fake/dir/clean_asphalt.jpg")
    assert record_texture_creators == ["create_asphalt_texture"]


def test_missing_normal_file_uses_flat_normal(record_texture_creators):
    texture_module.load_texture("/fake/dir/brick_nor_gl.png")
    assert record_texture_creators == ["get_flat_normal_texture"]


def test_missing_roughness_file_uses_default_roughness(record_texture_creators):
    texture_module.load_texture("/fake/dir/brick_rough.png")
    assert record_texture_creators == ["get_default_roughness_texture"]


def test_missing_unknown_file_falls_back_to_concrete(record_texture_creators):
    texture_module.load_texture("/fake/dir/mystery_material.png")
    assert record_texture_creators == ["create_concrete_texture"]


def test_empty_path_falls_back_to_concrete(record_texture_creators):
    texture_module.load_texture(None)
    assert record_texture_creators == ["create_concrete_texture"]
