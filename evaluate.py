"""Thin CLI wrapper around the installable ``molcompress`` package.

The implementation now lives in ``src/molcompress``. Run either:

    python evaluate.py --data qm9.csv
    molcompress-evaluate --data qm9.csv
"""

from __future__ import annotations

from molcompress import (  # re-exported for older scripts / tests
    FEATURE_NAMES,
    METHOD_PCA,
    METHOD_PINN,
    METHOD_RAW,
    compare,
    compute_metrics,
    load_dataset,
    save_artifacts,
    summary,
)
from molcompress.cli import main, parse_args
from molcompress.evaluation import demo_data

# Backward-compatible name used by the original smoke tests.
generate_synthetic_qm9 = demo_data


if __name__ == "__main__":
    main()
