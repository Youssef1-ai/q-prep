"""Command-line entry point: ``molcompress-evaluate``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import FEATURE_NAMES
from .evaluation import compare, features, load_dataset, save_artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="molcompress-evaluate",
        description="Train a physics-informed compressor and compare it with PCA.",
    )
    p.add_argument("--data", type=Path, default=Path("qm9.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=3)
    p.add_argument("--physics-weight", type=float, default=5.0)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--demo-samples", type=int, default=2400)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--clip-percentile", type=float, default=99.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df, label = load_dataset(
        args.data,
        demo_samples=args.demo_samples,
        seed=args.seed,
        max_samples=args.max_samples,
        clip_percentile=args.clip_percentile,
    )
    print(f"\nDataset: {label}")
    X = features(df)
    print(f"Train: {int(len(X) * (1 - args.test_size)):,} | Test: {int(len(X) * args.test_size):,}")

    result = compare(
        X,
        test_size=args.test_size,
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.hidden_dims),
        physics_weight=args.physics_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        verbose=args.verbose,
    )

    out_dir = save_artifacts(result, args.output_dir, label)
    print("\n================ PAPER RESULTS ================")
    print(result["metrics"].round(6).to_string(index=False))
    pca_v = result["metrics_pca"]["violation_mean_ev"]
    pinn_v = result["metrics_pinn"]["violation_mean_ev"]
    change = 100.0 * (pca_v - pinn_v) / pca_v if pca_v > 0 else 0.0
    print("----------------------------------------------")
    print(f"PCA -> PINN change in physical violation: {change:+.1f}%")
    print("NOTE: PCA is linear, so it preserves gap = lumo - homo by construction.")
    print(f"Acceptance threshold: {result['threshold']:.5f} eV")
    print(f"Artifacts: {out_dir.resolve()}")
    print("==============================================\n")


if __name__ == "__main__":
    main()
