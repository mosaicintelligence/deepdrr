from typing import TypeVar
from .renderable import Renderable
from .volume import Volume, MetalVolume
from .kwire import KWire
from .mesh import Mesh, LinearToolMesh
from .additive_density_field import AdditiveDensityField, validate_density_array

AnyVolume = TypeVar("AnyVolume", bound=Volume)

__all__ = [
    "Renderable",
    "Volume",
    "MetalVolume",
    "KWire",
    "AnyVolume",
    "Mesh",
    # Deliberately NOT a Renderable: it is added, not composited by the priority rule.
    # Passed as Projector(..., density_fields=[...]). Mosaic fork item 2.
    "AdditiveDensityField",
    "validate_density_array",
]
