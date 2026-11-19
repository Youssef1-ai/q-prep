# Virtual Molecular Screening Dashboard

Physics-informed dimensionality reduction of QM9 molecules: 11 descriptors → a
3D latent space, with the HOMO–LUMO reactivity law enforced **during training**
in physical units.

## Files

| File | Part | Purpose |
|------|------|---------|
| `src/molcompress/` | 1 | **Installable package.** Simple API: `Compressor`, `compare`, `load_dataset`, `save_artifacts` |
| `pyproject.toml` | 1 | Package metadata; builds `dist/molcompress-0.1.0-py3-none-any.whl` |
| `physics_compressor.py` | 1 | Backward-compatible shim re-exporting the package's core class |
| `evaluate.py` | 2 | Thin CLI wrapper around `molcompress.compare()` |
| `QM9_Comparison.ipynb` | 2 | **Colab-ready, executed notebook** with the PCA-vs-PINN comparison and plots |
| `app.py` | 3 | Streamlit dashboard: viability screener + judge-facing PCA vs. PINN comparison |
| `dist/` | — | Built wheel to upload to Google Colab |
| `requirements.txt` | — | Dependencies for running everything in one environment |
| `qm9.csv` | — | Place your QM9 CSV here (optional; demo data used if absent) |

## Quickstart

A ready-to-use `.venv` with all dependencies (CPU-only PyTorch) is included
in this folder. If you need to recreate it:

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Run the project:

```bash
source .venv/bin/activate
python evaluate.py                    # trains, evaluates, writes artifacts/
streamlit run app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`).

## Install the package by itself

```bash
pip install dist/molcompress-0.1.0-py3-none-any.whl
# or, from this folder: pip install -e .
```

Then use the simple, scikit-learn-style API:

```python
from molcompress import Compressor, load_dataset, features

df, label = load_dataset("qm9.csv")
X = features(df)

model = Compressor().fit(X)          # or fit(X_train)
Z = model.transform(X)               # (n, 3) latent coordinates
X_hat = model.inverse_transform(Z)   # (n, 11) physical units
violation = model.violation(X)       # |gap - (lumo - homo)| in eV
approved = model.approve(X)          # boolean viability flags
model.save("compressor.pt")
```

Older names still work: `encode = transform`, `decode = inverse_transform`,
`reconstruct`, `screen`, `load_dataset`, `compare`.

## Google Colab

Upload `QM9_Comparison.ipynb` to Colab. The first code cell asks you to upload
`dist/molcompress-0.1.0-py3-none-any.whl` and, optionally, `qm9.csv`. It installs
the package, runs the comparison, plots both 3D latent spaces, and saves
artifacts into `/content/artifacts`. The notebook already contains outputs from
the full 133,885-molecule run, so it can also just be viewed.

To use real QM9 data, place a CSV with columns
`A,B,C,alpha,homo,lumo,gap,r2,zpve,cv,u0` in this directory (or pass
`--data /path/to/qm9.csv`). When the electronic descriptors are in Hartree
(as in `/home/joseph/Downloads/qm9.csv`), the loader converts `homo`, `lumo`
and `gap` to eV automatically and clips the extreme `A/B/C` rotational
constants at the 99.9th percentile. Use `--max-samples N` to subsample and
`--clip-percentile 0` to disable clipping. The executed notebook
`QM9_Comparison.ipynb` runs the same pipeline on the full 133,885-molecule CSV
and regenerates the dashboard artefacts.

Useful options:

```bash
python evaluate.py --data qm9.csv --epochs 600 --physics-weight 2.0 --latent-dim 3
```

## Scientific design notes

1. **Differentiable inverse scaling.** The network trains on standardised
   features because QM9 descriptors span ~1e-3 (zpve) to ~1e3 (u0) Hartree.
   The penalty, however, is computed after unscaling:
   `x_phys = x_scaled * scale_buffer + mean_buffer`. registered torch buffers
   keep this operation inside the autograd graph; `StandardScaler.inverse_transform`
   would convert to NumPy and silently detach the gradient.

2. **Penalty on predictions.** The law is applied to the *reconstructed*
   `gap`, `lumo`, `homo`, not to the input row. Input rows that already obey the
   law give a near-zero penalty regardless of the model, so evaluating on
   inputs would measure nothing.

3. **Fair PCA baseline.** PCA is fitted on the same training split and the same
   standardisation as the autoencoder. Unscaled PCA would effectively select
   `u0` alone because of its magnitude.

4. **Two metrics.** Reconstruction MSE is reported in standardised units
   (primary, scale-independent) and physical units (per-feature detail). The
   Physical Violation Score is mean `|gap − (lumo − homo)|` of reconstructions
   in eV, plus std, p95 and max.

5. **Screening decision.** The dashboard takes the **maximum** of (a) the raw
   input inconsistency and (b) the reconstruction inconsistency. Without (a),
   a manually mistyped `gap` could be "repaired" by the decoder and falsely
   approved. The acceptance threshold is calibrated on training data (95th
   percentile × 1.5, floor 0.05 eV) and adjustable in the sidebar.

6. **Honest interpretation.** In QM9, `gap` is essentially defined as
   `lumo − homo` (to ~meV numerical precision). The physics term therefore
   acts as an electronic-structure consistency regulariser on the compressed
   representation, and becomes a genuine stability filter for user-edited or
   generated molecules outside the dataset manifold. It is **not** a substitute
   for DFT validation, and violation scores should not be presented as
   formation-energy or stability predictions.

## Important scientific caveat (read before presenting)

The law `gap = lumo − homo` is **linear**. PCA is a **linear** projector, so
if the source data satisfies the identity exactly (QM9 does, to ~meV), then
for any reconstruction `x̂ = P x`:

    ĝ − (l̂ − ĥ) = P(g − l + h) = 0

i.e. PCA preserves the law *by construction*, yielding a violation of ~0.
A nonlinear autoencoder has no such guarantee and will typically show a small
but nonzero violation (well below the 0.05 eV acceptance threshold in our
runs). Presenting the PINN as "beating PCA on this particular violation" on
clean QM9 data would be misleading.

What the experiment does show honestly:

- the PINN reconstructs the 11 descriptors substantially better than a
  3-component linear projection (lower MSE in standardised units);
- both methods satisfy the law to within a tiny fraction of the threshold on
  clean data;
- the viability screener's real value is the **raw-input check**, which catches
  user-edited or generated molecules whose stated gap contradicts their
  stated HOMO/LUMO levels — a case where a decoder could otherwise "repair"
  the inconsistency and falsely approve the molecule.

If a project requires a nonlinear physical law for a headline PINN-vs-PCA
violation result, use a genuinely nonlinear constraint (e.g. a DFT-derived
relation) rather than this linear identity.

## Artifacts written by `evaluate.py`

- `artifacts/compressor.pt` — weights, scaler statistics, threshold
- `artifacts/pca.joblib` — fitted PCA baseline
- `artifacts/test_embeddings.csv` — PCA/PINN latent coordinates, per-molecule
  violations and true features, consumed by the dashboard
- `artifacts/metrics.csv`, `artifacts/metrics.json` — headline results
