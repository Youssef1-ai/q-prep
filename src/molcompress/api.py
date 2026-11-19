"""Simple, scikit-learn-style public API for molecular compression.

    from molcompress import Compressor

    model = Compressor().fit(X_train)
    Z = model.transform(X)          # physical descriptors -> 3D latent
    X_hat = model.inverse_transform(Z)   # 3D latent -> physical descriptors
    approved = model.approve(X)     # viability flags
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from .core import FEATURE_NAMES, PhysicsInformedCompressor

PathLike = Union[str, Path]


class Compressor:
    """Physics-informed 3D compressor with familiar method names.

    The underlying network is a symmetric autoencoder trained with two
    objectives: reconstruction MSE and the HOMO-LUMO law
    ``gap = lumo - homo`` evaluated in eV space.

    Parameters
    ----------
    latent_dim:
        Size of the compressed representation (3 by default).
    physics_weight:
        Strength of the HOMO-LUMO penalty relative to reconstruction MSE.
    hidden_dims:
        Hidden-layer widths of the encoder/decoder.
    threshold:
        Optional override for the acceptance threshold (eV). If omitted,
        the threshold is calibrated from the training data during ``fit``.
    seed:
        Reproducibility seed.
    """

    def __init__(
        self,
        latent_dim: int = 3,
        physics_weight: float = 5.0,
        hidden_dims: Sequence[int] = (128, 64),
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self._model = PhysicsInformedCompressor(
            feature_names=FEATURE_NAMES,
            latent_dim=latent_dim,
            hidden_dims=tuple(hidden_dims),
            physics_weight=physics_weight,
            device=device,
            seed=seed,
        )
        if threshold is not None:
            self._model.violation_threshold.fill_(float(threshold))

    # ------------------------------------------------------------------ #
    # Exposed state
    # ------------------------------------------------------------------ #
    @property
    def model(self) -> PhysicsInformedCompressor:
        """The underlying torch module (advanced access)."""
        return self._model

    @property
    def scaler(self):
        """The fitted StandardScaler used for preprocessing."""
        return self._model.scaler

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._model.feature_names

    @property
    def threshold(self) -> float:
        """Acceptance threshold for the physical violation score (eV)."""
        return float(self._model.violation_threshold)

    @property
    def is_fitted(self) -> bool:
        return self._model.is_fitted

    @property
    def history_(self) -> Dict[str, list]:
        """Training loss history (compatibility with the core module)."""
        return self._model.history_

    @property
    def violation_threshold(self):
        """Alias of :attr:`threshold` (compatibility with the core module)."""
        return self._model.violation_threshold

    # ------------------------------------------------------------------ #
    # sklearn-style training / transformation
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: Any,
        epochs: int = 400,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "Compressor":
        """Fit the scaler and network. ``kwargs`` are forwarded to the trainer."""
        self._model.fit(
            X,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
            **kwargs,
        )
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Physical descriptors -> latent coordinates, shape ``(n, latent_dim)``."""
        return self._model.encode(X)

    def encode(self, X: Any) -> np.ndarray:
        """Alias of :meth:`transform`."""
        return self.transform(X)

    def fit_transform(self, X: Any, **fit_params: Any) -> np.ndarray:
        """Fit on ``X`` and immediately return its latent coordinates."""
        return self.fit(X, **fit_params).transform(X)

    def inverse_transform(self, Z: Any) -> np.ndarray:
        """Latent coordinates -> reconstructed descriptors in physical units."""
        return self._model.decode_tensor(Z).detach().cpu().numpy()

    def decode(self, Z: Any) -> np.ndarray:
        """Alias of :meth:`inverse_transform`."""
        return self.inverse_transform(Z)

    def reconstruct(self, X: Any) -> np.ndarray:
        """Physical descriptors -> reconstructed descriptors in physical units."""
        return self._model.reconstruct(X)

    # ------------------------------------------------------------------ #
    # Physical checks
    # ------------------------------------------------------------------ #
    def violation(self, X: Any) -> np.ndarray:
        """Per-molecule ``|gap - (lumo - homo)|`` of the reconstruction, in eV."""
        return self._model.predict_violation(X)

    def violation_score(self, X: Any) -> np.ndarray:
        """Alias of :meth:`violation`."""
        return self.violation(X)

    def score(self, X: Any) -> float:
        """Mean physical violation in eV (lower is better)."""
        return float(np.mean(self.violation(X)))

    def screen(self, X: Any, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Full screening report: violations, verdicts, latent codes, reconstruction."""
        return self._model.screen(X, threshold=threshold)

    def approve(self, X: Any, threshold: Optional[float] = None) -> np.ndarray:
        """Boolean array: ``True`` when a molecule passes the physical check."""
        return self.screen(X, threshold=threshold)["approved"]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: PathLike) -> Path:
        return self._model.save(path)

    @classmethod
    def load(cls, path: PathLike, device: Optional[str] = None) -> "Compressor":
        obj = cls()
        obj._model = PhysicsInformedCompressor.load(path, device=device)
        return obj

    def __repr__(self) -> str:
        return (
            f"Compressor(latent_dim={self._model.latent_dim}, "
            f"physics_weight={self._model.physics_weight}, fitted={self.is_fitted})"
        )
