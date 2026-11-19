"""Dataset loading, metrics and the PINN-vs-PCA comparison experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from .api import Compressor
from .core import FEATURE_NAMES, GAP_NAME, HOMO_NAME, LUMO_NAME

PathLike = Union[str, Path]
ELECTRONIC = [HOMO_NAME, LUMO_NAME, GAP_NAME]
HARTREE_TO_EV = 27.211386245988
METHOD_RAW = "Original (uncompressed)"
METHOD_PCA = "PCA (3 components)"
METHOD_PINN = "Physics-Informed AE (3D)"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def demo_data(n_samples: int = 2400, seed: int = 42) -> pd.DataFrame:
    """Physically consistent, nonlinear QM9-like demo data.

    Used only when no CSV is available. ``gap`` is built as ``lumo - homo``.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(3, 10, n_samples).astype(float)
    B = rng.integers(1, 6, n_samples).astype(float)
    C = rng.integers(0, 4, n_samples).astype(float)

    alpha = 12.0 + 7.5 * A + 2.6 * B + 9.0 * np.sin(0.6 * A) + rng.normal(0, 2.5, n_samples)
    homo = (
        -6.2 + 0.28 * A - 0.22 * B + 0.55 * np.sin(0.9 * A + 0.4 * B)
        + 0.30 * np.sqrt(C + 1.0) + rng.normal(0, 0.10, n_samples)
    )
    gap_true = (
        1.45 + 1.10 / (1.0 + 0.32 * A)
        + 0.80 * np.abs(np.sin(0.5 * A + 0.7 * B))
        + 0.25 * np.cos(1.1 * C) + rng.normal(0, 0.04, n_samples)
    )
    lumo = homo + gap_true
    gap = lumo - homo + rng.normal(0, 1e-4, n_samples)
    r2 = 18.0 + 6.0 * A + 0.35 * alpha + 4.0 * np.sin(0.8 * B) + rng.normal(0, 2.0, n_samples)
    zpve = 0.18 + 0.055 * A + 0.012 * B + 0.02 * np.cos(A) + rng.normal(0, 0.008, n_samples)
    cv = 12.0 + 3.2 * A + 1.4 * B + 2.0 * np.sin(0.5 * A * B) + rng.normal(0, 0.8, n_samples)
    u0 = -1100.0 * A - 95.0 * B + 50.0 * np.sin(0.4 * A) + rng.normal(0, 18.0, n_samples)

    return pd.DataFrame(
        {"A": A, "B": B, "C": C, "alpha": alpha, "homo": homo, "lumo": lumo,
         "gap": gap, "r2": r2, "zpve": zpve, "cv": cv, "u0": u0},
        columns=list(FEATURE_NAMES),
    )


def load_dataset(
    csv_path: PathLike,
    demo_samples: int = 2400,
    seed: int = 42,
    max_samples: int = 0,
    clip_percentile: float = 99.9,
) -> Tuple[pd.DataFrame, str]:
    """Load QM9 descriptors from CSV, with unit conversion and outlier clipping.

    Hartree-valued ``homo``/``lumo``/``gap`` columns are converted to eV. The
    heavy-tailed rotational constants A/B/C are clipped at ``clip_percentile``
    so a few extreme molecules cannot dominate StandardScaler or PCA.
    """
    csv_path = Path(csv_path)
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        df = df.loc[:, list(FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce")
        before = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        if len(df) < 50:
            raise ValueError(f"Only {len(df)} usable rows in {csv_path}.")

        notes = []
        if before != len(df):
            notes.append(f"dropped {before - len(df)} invalid rows")
        if float(np.nanmedian(np.abs(df[GAP_NAME]))) < 1.0:
            for col in ELECTRONIC:
                df.loc[:, col] = df[col] * HARTREE_TO_EV
            notes.append("converted homo/lumo/gap from Hartree to eV")
        if clip_percentile > 0:
            for col in ("A", "B", "C"):
                cap = float(np.percentile(df[col], clip_percentile))
                n = int((df[col] > cap).sum())
                if n:
                    df.loc[:, col] = df[col].clip(upper=cap)
                    notes.append(f"clipped {n} {col} outliers at {cap:.3f}")
        if 0 < max_samples < len(df):
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
            notes.append(f"sampled {max_samples} molecules (seed {seed})")

        label = f"QM9 CSV: {csv_path} (n={len(df)})"
        if notes:
            label += " -- " + "; ".join(notes)
        return df, label

    df = demo_data(demo_samples, seed=seed)
    return df, f"DEMO DATA (n={len(df)}) -- no CSV found; pass a real qm9.csv path"


def features(df: pd.DataFrame) -> np.ndarray:
    """Return the 11 model features as a float array in canonical order."""
    return df.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(x_true: np.ndarray, x_rec: np.ndarray, scaler: Any) -> Dict[str, Any]:
    """Reconstruction and physical-law metrics for one method."""
    x_true = np.asarray(x_true, dtype=np.float64)
    x_rec = np.asarray(x_rec, dtype=np.float64)

    diff_scaled = scaler.transform(x_true) - scaler.transform(x_rec)
    diff_phys = x_true - x_rec
    violation = np.abs(
        x_rec[:, FEATURE_NAMES.index(GAP_NAME)]
        - (x_rec[:, FEATURE_NAMES.index(LUMO_NAME)] - x_rec[:, FEATURE_NAMES.index(HOMO_NAME)])
    )
    electronic_idx = [FEATURE_NAMES.index(c) for c in ELECTRONIC]
    return {
        "recon_mse_standardized": float(np.mean(diff_scaled**2)),
        "recon_mse_physical": float(np.mean(diff_phys**2)),
        "electronics_mse_ev2": float(np.mean(diff_phys[:, electronic_idx] ** 2)),
        "violation_mean_ev": float(np.mean(violation)),
        "violation_std_ev": float(np.std(violation)),
        "violation_p95_ev": float(np.percentile(violation, 95)),
        "violation_max_ev": float(np.max(violation)),
        "per_feature_mse_physical": np.mean(diff_phys**2, axis=0),
        "violation_per_sample": violation,
    }


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def compare(
    X: Any,
    test_size: float = 0.2,
    latent_dim: int = 3,
    hidden_dims: Tuple[int, ...] = (128, 64),
    physics_weight: float = 5.0,
    epochs: int = 400,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = False,
    **fit_kwargs: Any,
) -> Dict[str, Any]:
    """Train the PINN and PCA on the same split and evaluate both on the test set.

    Returns a dictionary with ``model``, ``pca``, ``metrics`` (DataFrame),
    ``embeddings`` (per-molecule latent codes and violations) and raw arrays.
    """
    X = np.asarray(X, dtype=np.float64)
    X_train, X_test = train_test_split(
        X, test_size=test_size, random_state=seed, shuffle=True
    )

    model = Compressor(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        physics_weight=physics_weight,
        seed=seed,
    ).fit(
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=verbose,
        **fit_kwargs,
    )

    pca = PCA(n_components=latent_dim, random_state=seed).fit(
        model.scaler.transform(X_train)
    )
    Z_pca = pca.transform(model.scaler.transform(X_test))
    X_pca = model.scaler.inverse_transform(pca.inverse_transform(Z_pca))
    Z_pinn = model.transform(X_test)
    X_pinn = model.reconstruct(X_test)

    raw = compute_metrics(X_test, X_test, model.scaler)
    pca_m = compute_metrics(X_test, X_pca, model.scaler)
    pinn_m = compute_metrics(X_test, X_pinn, model.scaler)

    metrics_df = pd.DataFrame(
        [
            _metric_row(name, m)
            for name, m in (
                (METHOD_RAW, raw), (METHOD_PCA, pca_m), (METHOD_PINN, pinn_m)
            )
        ]
    )

    emb = pd.DataFrame(Z_pca, columns=[f"pca_{i + 1}" for i in range(latent_dim)])
    for i in range(Z_pinn.shape[1]):
        emb[f"pinn_{i + 1}"] = Z_pinn[:, i]
    emb.insert(0, "sample_id", np.arange(len(X_test)))
    emb["violation_raw"] = raw["violation_per_sample"]
    emb["violation_pca"] = pca_m["violation_per_sample"]
    emb["violation_pinn"] = pinn_m["violation_per_sample"]
    for j, name in enumerate(FEATURE_NAMES):
        emb[f"true_{name}"] = X_test[:, j]
    for name in ELECTRONIC:
        emb[f"recon_pinn_{name}"] = X_pinn[:, FEATURE_NAMES.index(name)]

    return {
        "model": model,
        "pca": pca,
        "X_train": X_train,
        "X_test": X_test,
        "Z_pca": Z_pca,
        "Z_pinn": Z_pinn,
        "X_pca": X_pca,
        "X_pinn": X_pinn,
        "metrics_raw": raw,
        "metrics_pca": pca_m,
        "metrics_pinn": pinn_m,
        "metrics": metrics_df,
        "embeddings": emb,
        "threshold": model.threshold,
    }


def _metric_row(name: str, m: Dict[str, Any]) -> Dict[str, float]:
    return {
        "method": name,
        "recon_mse_standardized": m["recon_mse_standardized"],
        "recon_mse_physical": m["recon_mse_physical"],
        "electronics_mse_ev2": m["electronics_mse_ev2"],
        "violation_mean_ev": m["violation_mean_ev"],
        "violation_std_ev": m["violation_std_ev"],
        "violation_p95_ev": m["violation_p95_ev"],
        "violation_max_ev": m["violation_max_ev"],
    }


def summary(result: Dict[str, Any], dataset_label: str = "") -> Dict[str, Any]:
    """Build the JSON summary used by the dashboard."""
    pca_v = result["metrics_pca"]["violation_mean_ev"]
    pinn_v = result["metrics_pinn"]["violation_mean_ev"]
    return {
        "dataset": dataset_label,
        "n_train": int(len(result["X_train"])),
        "n_test": int(len(result["X_test"])),
        "latent_dim": int(result["Z_pinn"].shape[1]),
        "physics_weight": float(result["model"].model.physics_weight),
        "epochs_run": len(result["model"].model.history_.get("train_loss", [])),
        "acceptance_threshold_ev": result["threshold"],
        "violation_reduction_percent": (
            100.0 * (pca_v - pinn_v) / pca_v if pca_v > 0 else 0.0
        ),
        "caveat": (
            "PCA is linear, so it preserves the linear identity "
            "gap = lumo - homo by construction; the PINN's advantage "
            "here is reconstruction fidelity."
        ),
        "metrics": {r["method"]: r for r in result["metrics"].to_dict("records")},
        "per_feature_mse_physical": {
            "feature": list(FEATURE_NAMES),
            METHOD_PCA: result["metrics_pca"]["per_feature_mse_physical"].tolist(),
            METHOD_PINN: result["metrics_pinn"]["per_feature_mse_physical"].tolist(),
        },
    }


def save_artifacts(result: Dict[str, Any], out_dir: PathLike, dataset_label: str = "") -> Path:
    """Save model, PCA, metrics and embeddings for the Streamlit app."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result["model"].save(out_dir / "compressor.pt")
    joblib.dump(result["pca"], out_dir / "pca.joblib")
    result["metrics"].to_csv(out_dir / "metrics.csv", index=False)
    result["embeddings"].to_csv(out_dir / "test_embeddings.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(summary(result, dataset_label), indent=2)
    )
    return out_dir
