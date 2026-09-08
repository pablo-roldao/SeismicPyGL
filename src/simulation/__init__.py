"""Módulo de Simulação: física sísmica e sistema de partículas."""

from .earthquake import EarthquakeSimulator
from .particles import ParticleSystem

__all__ = [
    "EarthquakeSimulator",
    "ParticleSystem",
]
