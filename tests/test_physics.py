"""Runtime sanity checks (no pytest required):  python tests/test_physics.py

Verifies the three claims that matter scientifically:
  1. unscaling keeps gradients attached;
  2. the physics penalty actually trains the encoder AND decoder;
  3. screening rejects a molecule whose gap contradicts lumo - homo;
  4. save/load reproduces predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physics_compressor import FEATURE_NAMES, PhysicsInformedCompressor  # noqa: E402


def main() -> None:
    from evaluate import generate_synthetic_qm9

    X_df = generate_synthetic_qm9(300, seed=7)
    model = PhysicsInformedCompressor(hidden_dims=(32, 16), physics_weight=2.0)
    model.fit(X_df, epochs=30, batch_size=64, verbose=False)

    # 1. differentiable inverse transform
    x = model._as_tensor(X_df.iloc[:5])
    scaled = model.scale_tensor(x).detach().requires_grad_(True)
    physical = model.to_physical_units(scaled)
    assert physical.requires_grad and physical.grad_fn is not None, "unscale detached the graph"
    physical.sum().backward()
    assert torch.isfinite(scaled.grad).all() and scaled.grad.abs().sum() > 0

    # 2. gradients reach both networks through the eV-space penalty
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model.physics_loss(X_df.iloc[:16])["total"]
    loss.backward()
    for layer in (model.encoder[0], model.decoder[-1]):
        grad = layer.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, (
            f"no gradient reached {layer}"
        )

    # 3. screener rejects an intentionally inconsistent molecule
    good = X_df.iloc[[0]].copy()
    result_good = model.screen(good)
    assert bool(result_good["approved"][0]), "clean QM9-like molecule was rejected"

    bad = good.copy()
    bad.loc[:, "gap"] += 2.0
    result_bad = model.screen(bad)
    assert not bool(result_bad["approved"][0]), "physically unstable molecule was approved"
    assert result_bad["effective_violation"][0] > result_good["effective_violation"][0]

    # 4. persistence round-trip
    path = Path("/tmp/opencode/compressor_smoke.pt")
    model.save(path)
    restored = PhysicsInformedCompressor.load(path)
    np.testing.assert_allclose(model.encode(good), restored.encode(good), atol=1e-5)
    np.testing.assert_allclose(model.reconstruct(good), restored.reconstruct(good), atol=1e-4)

    print("ALL SMOKE TESTS PASSED")
    print(f"latent shape: {model.encode(X_df.iloc[:3]).shape}")
    print(f"calibrated threshold: {float(model.violation_threshold):.4f} eV")


if __name__ == "__main__":
    main()
