"""molcompress — physics-informed 3D compression for QM9 molecules.

Quick start
-----------
    from molcompress import Compressor, load_dataset, compare

    model = Compressor().fit(X_train)       # X: (n, 11) QM9 descriptors
    Z = model.transform(X_test)             # -> (n, 3)
    X_hat = model.inverse_transform(Z)      # -> (n, 11) in physical units
    approved = model.approve(X_test)        # -> (n,) booleans
"""

from .api import Compressor
from .core import FEATURE_NAMES, PhysicsInformedCompressor
from .evaluation import (
    ELECTRONIC,
    HARTREE_TO_EV,
    METHOD_PCA,
    METHOD_PINN,
    METHOD_RAW,
    compare,
    compute_metrics,
    demo_data,
    features,
    load_dataset,
    save_artifacts,
    summary,
)

__version__ = "0.1.0"

__all__ = [
    "Compressor",
    "PhysicsInformedCompressor",
    "FEATURE_NAMES",
    "ELECTRONIC",
    "HARTREE_TO_EV",
    "METHOD_RAW",
    "METHOD_PCA",
    "METHOD_PINN",
    "compare",
    "compute_metrics",
    "demo_data",
    "features",
    "load_dataset",
    "save_artifacts",
    "summary",
    "__version__",
]
