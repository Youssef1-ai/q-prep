"""Part 1 -- Core algorithms for the Virtual Molecular Screening Dashboard.

PhysicsInformedCompressor
-------------------------
A PyTorch autoencoder that compresses QM9 molecules (11 descriptors) into a
3-dimensional latent space while enforcing the HOMO-LUMO reactivity law:

        gap  ==  lumo - homo

The important implementation detail is *where* the law is enforced.  The
network is trained on standardised inputs (StandardScaler), but the physical
penalty is evaluated after mapping the network's reconstruction back into
true physical units (eV for the electronic descriptors).  Crucially, this
inverse transform is implemented with torch tensor arithmetic on registered
buffers rather than with ``scaler.inverse_transform``: the latter converts to
NumPy and severs the autograd graph, so no physics gradient would ever reach
the encoder or decoder.

        x_phys = x_scaled * scale_buffer + mean_buffer   # differentiable

This single method is the difference between a real physics-informed model
and a post-hoc filter.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# QM9 descriptor order used throughout the project.
FEATURE_NAMES: Tuple[str, ...] = (
    "A", "B", "C", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0",
)
HOMO_NAME = "homo"
LUMO_NAME = "lumo"
GAP_NAME = "gap"

ArrayLike = Union[np.ndarray, pd.DataFrame, torch.Tensor, Sequence[float]]


class PhysicsInformedCompressor(nn.Module):
    """Physics-informed autoencoder for QM9 molecular descriptors.

    Parameters
    ----------
    feature_names:
        Column order of the input data.  Must contain ``homo``, ``lumo`` and
        ``gap``.
    latent_dim:
        Dimensionality of the compressed representation (3 for visualisation).
    hidden_dims:
        Widths of the hidden layers; encoder and decoder are symmetric.
    physics_weight:
        Multiplier applied to the physics penalty in the total loss.
    device:
        ``"cpu"`` / ``"cuda"`` / ``torch.device``.  Auto-detected if omitted.
    seed:
        Reproducibility seed for weight initialisation and shuffling.

    Attributes
    ----------
    scaler
        sklearn ``StandardScaler`` fitted on the TRAINING split only.
    scaler_mean, scaler_scale
        Torch buffers mirroring the scaler.  Used by :meth:`to_physical_units`
        so the inverse transform remains differentiable.
    violation_threshold
        Calibrated acceptance threshold in eV (95th percentile of training
        violations, inflated by 1.5x with a 0.05 eV floor).
    """

    def __init__(
        self,
        feature_names: Sequence[str] = FEATURE_NAMES,
        latent_dim: int = 3,
        hidden_dims: Sequence[int] = (128, 64),
        physics_weight: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.feature_names = tuple(feature_names)
        self.n_features = len(self.feature_names)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.physics_weight = float(physics_weight)
        self.seed = int(seed)
        self.dtype = torch.float32
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        missing = {HOMO_NAME, LUMO_NAME, GAP_NAME} - set(self.feature_names)
        if missing:
            raise ValueError(f"feature_names is missing required columns: {sorted(missing)}")
        self.homo_idx = self.feature_names.index(HOMO_NAME)
        self.lumo_idx = self.feature_names.index(LUMO_NAME)
        self.gap_idx = self.feature_names.index(GAP_NAME)

        torch.manual_seed(self.seed)
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # Scaler state as buffers: moves with .to(device) and is saved in
        # state_dict, while staying inside the autograd graph during use.
        self.register_buffer("scaler_mean", torch.zeros(self.n_features, dtype=self.dtype))
        self.register_buffer("scaler_scale", torch.ones(self.n_features, dtype=self.dtype))
        self.register_buffer("violation_threshold", torch.tensor(0.05, dtype=self.dtype))

        self.scaler = StandardScaler()
        self.is_fitted = False
        self.history_: Dict[str, list] = {"train_loss": [], "val_loss": []}
        self.to(self.device)

    # ------------------------------------------------------------------ #
    # Architecture
    # ------------------------------------------------------------------ #
    def _build_encoder(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = self.n_features
        for width in self.hidden_dims:
            layers += [nn.Linear(in_dim, width), nn.GELU()]
            in_dim = width
        layers.append(nn.Linear(in_dim, self.latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = self.latent_dim
        for width in reversed(self.hidden_dims):
            layers += [nn.Linear(in_dim, width), nn.GELU()]
            in_dim = width
        layers.append(nn.Linear(in_dim, self.n_features))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------ #
    # Data conversion helpers
    # ------------------------------------------------------------------ #
    def _to_numpy(self, X: ArrayLike) -> np.ndarray:
        """Return a 2-D float64 array in ``self.feature_names`` order."""
        if isinstance(X, torch.Tensor):
            arr = X.detach().cpu().numpy().astype(np.float64)
        elif isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            arr = X.loc[:, list(self.feature_names)].to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features {list(self.feature_names)}, "
                f"got {arr.shape[1]}."
            )
        if not np.isfinite(arr).all():
            raise ValueError("Input contains NaN or infinite values.")
        # Own the data: pandas slices can be read-only, which makes torch warn.
        return np.array(arr, dtype=np.float64, copy=True)

    def _as_tensor(self, X: ArrayLike) -> torch.Tensor:
        """Convert raw PHYSICAL inputs to a float32 tensor on ``self.device``."""
        if isinstance(X, torch.Tensor):
            t = X.to(device=self.device, dtype=self.dtype)
        else:
            t = torch.as_tensor(
                self._to_numpy(X), dtype=self.dtype, device=self.device
            )
        if t.ndim == 1:
            t = t.unsqueeze(0)
        if t.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {t.shape[-1]}."
            )
        return t

    def _sync_buffers(self) -> None:
        """Mirror the fitted sklearn scaler into torch buffers."""
        self.scaler_mean.copy_(
            torch.as_tensor(self.scaler.mean_, dtype=self.dtype, device=self.device)
        )
        self.scaler_scale.copy_(
            torch.as_tensor(self.scaler.scale_, dtype=self.dtype, device=self.device)
        )

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit(...) or .load(...) first.")

    # ------------------------------------------------------------------ #
    # Differentiable scaling -- the heart of the physics-informed loss
    # ------------------------------------------------------------------ #
    def scale_tensor(self, x_physical: torch.Tensor) -> torch.Tensor:
        """Standardise a tensor of physical descriptors (dataset statistics)."""
        mean = self.scaler_mean.to(device=x_physical.device, dtype=x_physical.dtype)
        scale = self.scaler_scale.to(device=x_physical.device, dtype=x_physical.dtype)
        return (x_physical - mean) / scale

    def to_physical_units(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """Inverse-standardise WITHOUT breaking the autograd graph.

        ``StandardScaler.inverse_transform`` routes through NumPy, which
        detaches the tensor.  Multiplying by registered buffers instead keeps
        d(loss)/d(weights) intact, so the HOMO-LUMO law genuinely trains the
        network in eV space rather than merely being reported after the fact.
        """
        mean = self.scaler_mean.to(device=x_scaled.device, dtype=x_scaled.dtype)
        scale = self.scaler_scale.to(device=x_scaled.device, dtype=x_scaled.dtype)
        return x_scaled * scale + mean  # autograd preserved

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """Map STANDARDISED inputs to a STANDARDISED reconstruction."""
        return self.decoder(self.encoder(x_scaled))

    # ------------------------------------------------------------------ #
    # Physics
    # ------------------------------------------------------------------ #
    def _violation_physical(self, x_physical: torch.Tensor) -> torch.Tensor:
        """Per-sample |gap - (lumo - homo)| in eV, differentiable."""
        gap = x_physical[:, self.gap_idx]
        delta = x_physical[:, self.lumo_idx] - x_physical[:, self.homo_idx]
        return (gap - delta).abs()

    def _violation_from_scaled(
        self, x_scaled: torch.Tensor, use_reconstruction: bool = True
    ) -> torch.Tensor:
        phys = (
            self.to_physical_units(self.forward(x_scaled))
            if use_reconstruction
            else self.to_physical_units(x_scaled)
        )
        return self._violation_physical(phys)

    def physics_loss(self, X: ArrayLike, already_scaled: bool = False) -> Dict[str, torch.Tensor]:
        """Compute the composite training objective.

        Returns a dict with ``total``, ``reconstruction_mse`` (standardised
        space) and ``physics_mse`` (physical eV^2 space).  Both terms remain
        differentiable through :meth:`to_physical_units`.
        """
        x = self._as_tensor(X)  # raw physical units
        x_scaled = x if already_scaled else self.scale_tensor(x)

        x_rec_scaled = self.forward(x_scaled)
        recon_mse = F.mse_loss(x_rec_scaled, x_scaled)

        # eV-space penalty on the PREDICTED electronic descriptors.
        x_rec_phys = self.to_physical_units(x_rec_scaled)
        physics_mse = F.mse_loss(
            x_rec_phys[:, self.gap_idx],
            x_rec_phys[:, self.lumo_idx] - x_rec_phys[:, self.homo_idx],
        )

        return {
            "total": recon_mse + self.physics_weight * physics_mse,
            "reconstruction_mse": recon_mse,
            "physics_mse": physics_mse,
        }

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: ArrayLike,
        epochs: int = 400,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        val_fraction: float = 0.1,
        patience: int = 40,
        verbose: bool = False,
    ) -> "PhysicsInformedCompressor":
        """Fit the scaler and the physics-informed autoencoder on ``X``.

        Only the training split is ever used for fitting; the holdout split is
        used exclusively for early stopping.
        """
        X_np = self._to_numpy(X)
        if len(X_np) < 4:
            raise ValueError("Need at least 4 molecules to train.")

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(X_np))
        n_val = int(len(X_np) * val_fraction) if len(X_np) >= 20 else 0
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        if len(train_idx) == 0:
            train_idx, val_idx = perm, perm[:0]

        # Fit preprocessing on the inner training split only, so the
        # early-stopping holdout never informs the scaler statistics.
        self.scaler.fit(X_np[train_idx])
        self._sync_buffers()

        X_train = torch.as_tensor(X_np[train_idx], dtype=self.dtype, device=self.device)
        X_val = (
            torch.as_tensor(X_np[val_idx], dtype=self.dtype, device=self.device)
            if len(val_idx) > 0
            else None
        )
        train_s = self.scale_tensor(X_train)

        g = torch.Generator()
        g.manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_s),
            batch_size=max(1, min(batch_size, len(train_s))),
            shuffle=True,
            generator=g,
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_loss = float("inf")
        best_state: Dict[str, torch.Tensor] = {}
        stale = 0
        self.history_ = {"train_loss": [], "val_loss": []}

        self.train()
        for epoch in range(1, int(epochs) + 1):
            self.train()
            epoch_loss = 0.0
            n_seen = 0
            for (batch,) in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = self.physics_loss(batch, already_scaled=True)["total"]
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(batch)
                n_seen += len(batch)
            train_loss = epoch_loss / max(n_seen, 1)

            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = float(
                        self.physics_loss(self.scale_tensor(X_val), already_scaled=True)[
                            "total"
                        ]
                    )
            else:
                val_loss = train_loss

            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)

            track = val_loss if X_val is not None else train_loss
            if track < best_loss - 1e-6:
                best_loss = track
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                stale = 0
            else:
                stale += 1

            if verbose and (epoch == 1 or epoch % 25 == 0 or stale == patience):
                print(
                    f"epoch {epoch:4d}/{epochs} | train {train_loss:.6f} "
                    f"| val {val_loss:.6f} | best {best_loss:.6f}"
                )
            if stale >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

        if best_state:
            self.load_state_dict(best_state)

        # Acceptance threshold calibrated on training data only.
        self.eval()
        with torch.no_grad():
            v_rec = self._violation_from_scaled(self.scale_tensor(X_train), True)
            v_raw = self._violation_physical(X_train)
            calib = torch.maximum(v_rec, v_raw).cpu().numpy()
        threshold = max(float(np.quantile(calib, 0.95)) * 1.5, 0.05)
        self.violation_threshold.fill_(threshold)

        self.is_fitted = True
        self.eval()
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def encode_tensor(self, X: ArrayLike) -> torch.Tensor:
        """Physical descriptors -> 3-D latent tensor."""
        self._check_fitted()
        x = self._as_tensor(X)
        return self.encoder(self.scale_tensor(x))

    def encode(self, X: ArrayLike) -> np.ndarray:
        """Physical descriptors -> latent NumPy array of shape (n, latent_dim)."""
        self._check_fitted()
        with torch.no_grad():
            return self.encode_tensor(X).cpu().numpy()

    def compress(self, X: ArrayLike) -> np.ndarray:
        """Alias for :meth:`encode`."""
        return self.encode(X)

    def decode_tensor(self, z: torch.Tensor) -> torch.Tensor:
        """Latent tensor -> reconstruction in PHYSICAL units (differentiable)."""
        self._check_fitted()
        if isinstance(z, np.ndarray):
            z = torch.as_tensor(z, dtype=self.dtype, device=self.device)
        z = z.to(device=self.device, dtype=self.dtype)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return self.to_physical_units(self.decoder(z))

    def reconstruction_tensor(self, X: ArrayLike) -> torch.Tensor:
        """Physical descriptors -> reconstructed descriptors in PHYSICAL units."""
        self._check_fitted()
        x = self._as_tensor(X)
        return self.to_physical_units(self.forward(self.scale_tensor(x)))

    def reconstruct(self, X: ArrayLike) -> np.ndarray:
        """Physical descriptors -> reconstruction in PHYSICAL units (NumPy)."""
        self._check_fitted()
        with torch.no_grad():
            return self.reconstruction_tensor(X).cpu().numpy()

    def predict_violation(
        self, X: ArrayLike, use_reconstruction: bool = True
    ) -> np.ndarray:
        """Per-molecule |gap - (lumo - homo)| in eV."""
        self._check_fitted()
        x = self._as_tensor(X)
        with torch.no_grad():
            v = (
                self._violation_from_scaled(self.scale_tensor(x), use_reconstruction)
                if use_reconstruction
                else self._violation_physical(x)
            )
        return v.cpu().numpy()

    def screen(
        self, X: ArrayLike, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Full viability screen for one or many molecules.

        The effective violation is the WORST of (a) the inconsistency of the
        raw user input and (b) the inconsistency of the model reconstruction.
        Checking only (b) would let a model "repair" a manually mistyped gap
        and incorrectly approve an unstable molecule.
        """
        self._check_fitted()
        x = self._as_tensor(X)
        thr = float(threshold) if threshold is not None else float(self.violation_threshold)
        with torch.no_grad():
            x_scaled = self.scale_tensor(x)
            z = self.encoder(x_scaled)
            raw_v = self._violation_physical(x)
            rec_v = self._violation_from_scaled(x_scaled, True)
            eff = torch.maximum(raw_v, rec_v)
            rec_phys = self.to_physical_units(self.decoder(z))
        return {
            "raw_violation": raw_v.cpu().numpy(),
            "reconstruction_violation": rec_v.cpu().numpy(),
            "effective_violation": eff.cpu().numpy(),
            "approved": (eff <= thr).cpu().numpy(),
            "threshold": thr,
            "latent": z.cpu().numpy(),
            "reconstruction": rec_phys.cpu().numpy(),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> Path:
        """Save configuration, weights, scaler statistics and threshold."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "feature_names": list(self.feature_names),
                "latent_dim": self.latent_dim,
                "hidden_dims": list(self.hidden_dims),
                "physics_weight": self.physics_weight,
                "seed": self.seed,
            },
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "scaler": {
                "mean_": self.scaler.mean_.astype(np.float64),
                "scale_": self.scaler.scale_.astype(np.float64),
                "var_": self.scaler.var_.astype(np.float64),
                "n_samples_seen_": int(self.scaler.n_samples_seen_),
            },
            "threshold": float(self.violation_threshold),
            "history": copy.deepcopy(self.history_),
            "is_fitted": bool(self.is_fitted),
        }
        torch.save(payload, str(path))
        return path

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "PhysicsInformedCompressor":
        """Restore a model saved with :meth:`save`.

        ``weights_only=False`` is required because the checkpoint carries
        NumPy scaler arrays as well as tensors.  Only load trusted files.
        """
        device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        payload = torch.load(str(path), map_location=device, weights_only=False)
        obj = cls(**payload["config"], device=device)
        obj.load_state_dict(payload["state_dict"])

        sc = StandardScaler()
        s = payload["scaler"]
        sc.mean_ = np.asarray(s["mean_"], dtype=np.float64)
        sc.scale_ = np.asarray(s["scale_"], dtype=np.float64)
        sc.var_ = np.asarray(s["var_"], dtype=np.float64)
        sc.n_samples_seen_ = int(s["n_samples_seen_"])
        sc.n_features_in_ = obj.n_features
        obj.scaler = sc

        obj.violation_threshold.fill_(float(payload["threshold"]))
        obj.history_ = payload.get("history", {"train_loss": [], "val_loss": []})
        obj.is_fitted = bool(payload.get("is_fitted", True))
        obj.to(device)
        obj.eval()
        return obj
