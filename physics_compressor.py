"""Backward-compatible shim.

The algorithm now lives in the installable ``molcompress`` package:

    from molcompress import Compressor

This module is kept so older scripts and tests that import
``physics_compressor`` continue to work unchanged.
"""

from molcompress.core import (
    FEATURE_NAMES,
    GAP_NAME,
    HOMO_NAME,
    LUMO_NAME,
    PhysicsInformedCompressor,
)

__all__ = [
    "PhysicsInformedCompressor",
    "FEATURE_NAMES",
    "HOMO_NAME",
    "LUMO_NAME",
    "GAP_NAME",
]
