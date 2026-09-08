"""Módulo de Cenário e Mundo 3D do SeismicPyGL."""

from .shared import (
    get_pbr_set, get_shared_cube_mesh, get_shared_cone_mesh,
    get_shared_cylinder_mesh, get_shared_pyramid_mesh, get_shared_gable_mesh,
    get_concrete_texture, get_grass_texture, get_asphalt_texture,
    reset_shared_resources
)
from .ground import Ground
from .building import Building, create_house, BuildingDebris
from .street import Street
from .light_pole import LightPole, LampPost
from .nature import Tree, generate_forest
from .mountain import Mountain, RockDebris
from .village import generate_village, generate_city
from .debris_renderer import DebrisRenderer

__all__ = [
    "get_pbr_set", "get_shared_cube_mesh", "get_shared_cone_mesh",
    "get_shared_cylinder_mesh", "get_shared_pyramid_mesh", "get_shared_gable_mesh",
    "get_concrete_texture", "get_grass_texture", "get_asphalt_texture",
    "reset_shared_resources",
    "Ground",
    "Building", "create_house", "BuildingDebris",
    "Street",
    "LightPole", "LampPost",
    "Tree", "generate_forest",
    "Mountain", "RockDebris",
    "generate_village", "generate_city",
    "DebrisRenderer",
]
