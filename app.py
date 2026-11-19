"""Q-PREP: A PIDR-QML System for Drug Screening.

Production Streamlit dashboard wired directly to the trained PyTorch
physics-informed autoencoder in ``artifacts/compressor.pt``. The model weights
and architecture are never retrained or modified here.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
# Works from a fresh clone / Streamlit Cloud where the package is not installed.
sys.path.insert(0, str(ROOT / "src"))

from molcompress import Compressor, FEATURE_NAMES, HARTREE_TO_EV

ART = ROOT / "artifacts"
MODEL_PATH = ART / "compressor.pt"
EMB_PATH = ART / "test_embeddings.csv"
METRICS_PATH = ART / "metrics.csv"

APPROVED_TEXT = "Approved: Physically Stable - Ready for Trials"
REJECTED_TEXT = "Rejected: Physical Instability Detected"
SOURCE_SAMPLE = "QM9 test sample (works out of the box)"
SOURCE_UPLOAD = "Upload CSV"

UNITS: Dict[str, str] = {
    "A": "rotational constant A (GHz)",
    "B": "rotational constant B (GHz)",
    "C": "rotational constant C (GHz)",
    "alpha": "isotropic polarizability (Å³)",
    "homo": "HOMO (eV)",
    "lumo": "LUMO (eV)",
    "gap": "HOMO-LUMO gap (eV)",
    "r2": "rotational radius (Å²)",
    "zpve": "zero-point energy (eV)",
    "cv": "heat capacity (cal/mol/K)",
    "u0": "internal energy (Hartree)",
}
MAX_SCREEN_ROWS = 2000
MAX_PLOT_ROWS = 3000

st.set_page_config(
    page_title="Q-PREP: PIDR-QML Drug Screening",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; }
        .stTabs [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 650; }
        div[data-testid="stMetricValue"] { font-size: 1.5rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.86rem; }
        .qcard {
            padding: 0.9rem 1rem;
            border-radius: 0.75rem;
            border-left: 6px solid;
            margin-bottom: 0.6rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }
        .qcard-title { font-weight: 700; font-size: 1.02rem; }
        .qcard-status { font-weight: 750; font-size: 0.98rem; margin-top: 0.25rem; }
        .qcard-meta { font-size: 0.84rem; opacity: 0.88; margin-top: 0.35rem; }
        .qhero {
            padding: 1.1rem 1.2rem;
            border-radius: 0.9rem;
            background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 100%);
            border: 1px solid #dbe3ff;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Artifact loading -- never retrains, never edits the saved weights
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained Q-PREP model ...")
def load_model(path: str) -> Compressor:
    return Compressor.load(path)


@st.cache_data(show_spinner=False)
def read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def require_artifacts() -> None:
    missing = [p.name for p in (MODEL_PATH, EMB_PATH) if not p.is_file()]
    if missing:
        st.error("Missing required artifacts: " + ", ".join(missing))
        st.code("python evaluate.py", language="bash")
        st.stop()


require_artifacts()
model = load_model(str(MODEL_PATH))
emb = read_csv(str(EMB_PATH), EMB_PATH.stat().st_mtime)
reference_metrics = (
    read_csv(str(METRICS_PATH), METRICS_PATH.stat().st_mtime)
    if METRICS_PATH.is_file()
    else pd.DataFrame()
)

H_IDX = FEATURE_NAMES.index("homo")
L_IDX = FEATURE_NAMES.index("lumo")
G_IDX = FEATURE_NAMES.index("gap")
ELECTRONIC = ("homo", "lumo", "gap")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sample_molecules(n: int) -> pd.DataFrame:
    """Build a DataFrame of real QM9 test molecules from saved embeddings."""
    n = max(1, min(int(n), len(emb)))
    frame = pd.DataFrame(
        {name: emb[f"true_{name}"].to_numpy()[:n] for name in FEATURE_NAMES}
    )
    frame.insert(0, "sample_id", emb["sample_id"].to_numpy()[:n])
    return frame


def normalise_upload(uploaded) -> Optional[pd.DataFrame]:
    """Validate and standardise an uploaded molecule CSV."""
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return None

    missing = [c for c in FEATURE_NAMES if c not in raw.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required columns: {missing}")
        return None

    df = raw.loc[:, list(FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce")
    if float(np.nanmedian(np.abs(df["gap"]))) < 1.0:
        for col in ELECTRONIC:
            df.loc[:, col] = df[col] * HARTREE_TO_EV
        st.info("Converted HOMO/LUMO/GAP from Hartree to eV.")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if df.empty:
        st.error("No valid numeric molecule rows remain after cleaning.")
        return None
    if len(df) > MAX_SCREEN_ROWS:
        df = df.sample(n=MAX_SCREEN_ROWS, random_state=42).reset_index(drop=True)
        st.info(f"Showing {MAX_SCREEN_ROWS:,} randomly selected molecules for performance.")
    return df


def qprep_engine(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the REAL encoder -> decoder and return latent, reconstruction, error.

    The physics score requested by the product spec is computed exactly as
    |(LUMO - HOMO) - GAP| from the DECODER OUTPUT in physical units.
    """
    Z = model.transform(X)                      # real PyTorch encoder
    reconstruction = model.inverse_transform(Z)  # real PyTorch decoder
    error = np.abs(
        (reconstruction[:, L_IDX] - reconstruction[:, H_IDX])
        - reconstruction[:, G_IDX]
    )
    return Z, reconstruction, error


def result_card(row: pd.Series, threshold: float) -> str:
    approved = float(row["error"]) < threshold
    color = "#16a34a" if approved else "#dc2626"
    bg = "#ecfdf5" if approved else "#fef2f2"
    status = APPROVED_TEXT if approved else REJECTED_TEXT
    sid = row.get("sample_id", "—")
    return f"""
    <div class="qcard" style="border-left-color:{color}; background:{bg};">
        <div class="qcard-title">Molecule #{sid}</div>
        <div class="qcard-status" style="color:{color};">{status}</div>
        <div class="qcard-meta">
            Physics violation score: <b>{float(row['error']):.5f} eV</b>
            &nbsp;·&nbsp; threshold {threshold:.3f} eV
        </div>
        <div class="qcard-meta">
            HOMO {float(row['recon_homo']):.3f} eV &nbsp;·&nbsp;
            LUMO {float(row['recon_lumo']):.3f} eV &nbsp;·&nbsp;
            GAP {float(row['recon_gap']):.3f} eV
        </div>
    </div>
    """


def set_active_data(df: pd.DataFrame) -> None:
    """Replace the active molecules and drop every result derived from old data."""
    st.session_state.qprep_data = df
    st.session_state.qprep_results = None
    st.session_state.qprep_latent = None
    st.session_state.validation = None


def ensure_state() -> None:
    if "qprep_data" not in st.session_state:
        st.session_state.qprep_data = sample_molecules(25)
    if "qprep_results" not in st.session_state:
        st.session_state.qprep_results = None
    if "qprep_latent" not in st.session_state:
        st.session_state.qprep_latent = None
    if "validation" not in st.session_state:
        st.session_state.validation = None
    if "qprep_source" not in st.session_state:
        st.session_state.qprep_source = SOURCE_SAMPLE


ensure_state()


# --------------------------------------------------------------------------- #
# Professional sidebar
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## ⚛️ Q-PREP Control Center")
st.sidebar.caption("Physics-Informed Dimensionality Reduction · Quantum Machine Learning")
st.sidebar.markdown("---")

st.sidebar.markdown("### System")
st.sidebar.metric("Model", "PIDR-QML Autoencoder")
st.sidebar.metric("Latent dimensions", 3)
st.sidebar.metric("Input descriptors", len(FEATURE_NAMES))
st.sidebar.metric("Physics weight", float(model.model.physics_weight))
st.sidebar.metric("Epochs trained", len(model.history_.get("train_loss", [])))
if len(reference_metrics):
    pinn_row = reference_metrics.loc[
        reference_metrics["method"] == "Physics-Informed AE (3D)"
    ]
    if len(pinn_row):
        st.sidebar.metric(
            "QM9 reference MSE",
            f"{float(pinn_row.iloc[0]['recon_mse_standardized']):.4f}",
        )

st.sidebar.markdown("### Decision policy")
threshold = st.sidebar.slider(
    "Stability threshold (eV)",
    min_value=0.001, max_value=0.50, value=0.05, step=0.001,
    help="Molecules with |(LUMO − HOMO) − GAP| below this value are approved.",
)
st.sidebar.caption("Rule: error < threshold → physically stable.")
st.sidebar.markdown("### View options")
sample_n = st.sidebar.selectbox(
    "QM9 sample size", options=[1, 5, 10, 25, 50, 100, 250], index=3
)
if st.sidebar.button("↺ Reset to QM9 sample"):
    set_active_data(sample_molecules(sample_n))
    st.session_state.qprep_source = SOURCE_SAMPLE
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Model weights are loaded from `artifacts/compressor.pt` and are never retrained in this app.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="qhero">'
    '<h1 style="margin:0 0 0.35rem 0;">Q-PREP</h1>'
    '<h3 style="margin:0;color:#4338ca;">A PIDR-QML System for Drug Screening</h3>'
    '<p style="margin:0.5rem 0 0 0;">Compress 11 molecular descriptors into a '
    '3D quantum-ready latent space, reconstruct them through a trained physics-informed '
    'autoencoder, and screen for HOMO-LUMO instability before real-world trials.</p>'
    '</div>',
    unsafe_allow_html=True,
)

tab_screener, tab_space, tab_validation = st.tabs(
    [
        "🧬 Virtual Screener",
        "🌌 3D Quantum Space",
        "🔬 Scientific Validation",
    ]
)


# =========================================================================== #
# TAB 1 -- Virtual Screener
# =========================================================================== #
with tab_screener:
    st.header("Virtual Screener — Real-world Application")
    st.write(
        "Upload a CSV of new molecules or use a ready-made sample from the QM9 test "
        "set. Press **Run Q-PREP Engine** to pass every molecule through the actual "
        "trained encoder and decoder."
    )

    source = st.radio(
        "Input source",
        [SOURCE_SAMPLE, SOURCE_UPLOAD],
        horizontal=True,
    )

    if source != st.session_state.qprep_source:
        st.session_state.qprep_source = source
        st.session_state.qprep_results = None
        st.session_state.qprep_latent = None
        st.session_state.validation = None
        if source == SOURCE_SAMPLE:
            st.session_state.qprep_data = sample_molecules(sample_n)

    if source == SOURCE_UPLOAD:
        uploaded = st.file_uploader(
            "Upload molecule CSV", type=["csv"],
            help="Must contain A, B, C, alpha, homo, lumo, gap, r2, zpve, cv, u0.",
        )
        if uploaded is not None:
            cleaned = normalise_upload(uploaded)
            if cleaned is not None:
                set_active_data(cleaned)
                st.success(f"Loaded {len(cleaned):,} molecules from {uploaded.name}.")
    else:
        new_sample = sample_molecules(sample_n)
        if not st.session_state.qprep_data.equals(new_sample):
            set_active_data(new_sample)

    active = st.session_state.qprep_data.copy()
    st.caption(f"Active dataset: {len(active):,} molecules")
    st.dataframe(active.head(10), width="stretch", hide_index=True)

    st.markdown("---")
    run = st.button("🚀 Run Q-PREP Engine", type="primary")
    if run:
        with st.spinner("Running real PyTorch encoder and decoder ..."):
            X = active.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
            Z, rec, error = qprep_engine(X)
        results = active.copy()
        if "sample_id" not in results.columns:
            results.insert(0, "sample_id", np.arange(len(active)))
        results["error"] = error
        for name in ELECTRONIC:
            results[f"recon_{name}"] = rec[:, FEATURE_NAMES.index(name)]
        st.session_state.qprep_results = results
        st.session_state.qprep_latent = Z

    results = st.session_state.qprep_results
    if results is not None:
        n_ok = int((results["error"] < threshold).sum())
        n_bad = len(results) - n_ok
        st.subheader("Screening results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Molecules screened", len(results))
        c2.metric("Approved", n_ok)
        c3.metric("Rejected", n_bad)

        if n_bad == 0:
            st.success(APPROVED_TEXT)
        else:
            st.error(f"{n_bad} molecule(s): {REJECTED_TEXT}")

        st.markdown("#### Molecule cards")
        card_columns = st.columns(2)
        for i, (_, row) in enumerate(results.head(10).iterrows()):
            with card_columns[i % 2]:
                st.markdown(result_card(row, threshold), unsafe_allow_html=True)
        if len(results) > 10:
            st.caption(f"Showing the first 10 cards; all {len(results):,} results are in the table below.")

        st.markdown("#### Full results")
        st.dataframe(
            results[["sample_id", "error", "recon_homo", "recon_lumo", "recon_gap"]]
            .round(6),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "⬇ Download screening results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="qprep_screening_results.csv",
            mime="text/csv",
        )
    else:
        st.info("Press **Run Q-PREP Engine** to screen the active molecules.")


# =========================================================================== #
# TAB 2 -- 3D Quantum Space
# =========================================================================== #
with tab_space:
    st.header("3D Quantum Space — Dimensionality Reduction")
    st.write(
        "The real PyTorch encoder maps each 11-descriptor molecule into a "
        "three-dimensional latent coordinate. Rotate, zoom and hover to inspect "
        "the chemical properties preserved in the compressed representation."
    )

    X_active = active.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    Z_active = st.session_state.qprep_latent
    if Z_active is None or len(Z_active) != len(active):
        Z_active = model.transform(X_active)

    plot_df = pd.DataFrame(Z_active, columns=["z1", "z2", "z3"])
    for j, name in enumerate(FEATURE_NAMES):
        plot_df[name] = X_active[:, j]
    plot_df.insert(0, "sample_id", active.get("sample_id", pd.Series(range(len(active)))).to_numpy())

    results = st.session_state.qprep_results
    if results is not None and len(results) == len(plot_df):
        plot_df["Physics violation (eV)"] = results["error"].to_numpy()
        color_col = "Physics violation (eV)"
    else:
        color_col = "gap"

    if len(plot_df) > MAX_PLOT_ROWS:
        plot_df = plot_df.sample(n=MAX_PLOT_ROWS, random_state=42)
        st.info(f"Plotting {MAX_PLOT_ROWS:,} randomly selected points for interactivity.")

    hover_cols = ["sample_id"] + list(FEATURE_NAMES)
    if color_col in plot_df.columns and color_col not in hover_cols:
        hover_cols.append(color_col)

    fig = px.scatter_3d(
        plot_df,
        x="z1", y="z2", z="z3",
        color=color_col,
        hover_data=hover_cols,
        labels={**UNITS, "z1": "Latent z₁", "z2": "Latent z₂", "z3": "Latent z₃",
                color_col: "Energy / violation (eV)"},
        color_continuous_scale="RdYlGn_r",
        opacity=0.88,
    )
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis_title="z₁", yaxis_title="z₂", zaxis_title="z₃",
            aspectmode="cube",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
        **Why "Quantum-Ready"?** Three latent dimensions fit naturally onto three
        qubits. Each molecule becomes a compact state preparation target
        `|ψ⟩ = α|000⟩ + … + η|111⟩`, making the representation directly usable
        for near-term **NISQ** quantum kernels and variational algorithms.
        """
    )


# =========================================================================== #
# TAB 3 -- Scientific Validation
# =========================================================================== #
with tab_validation:
    st.header("Scientific Validation — Q-PREP vs. PCA")
    st.write(
        "A standard `PCA(n_components=3)` baseline is fitted on the same data, then "
        "inverse-transformed back to 11 descriptors. Both reconstructions are scored "
        "with the identical HOMO-LUMO physical-law error."
    )

    val_source = st.selectbox(
        "Validation dataset",
        ["QM9 reference test set (full)", "Currently screened molecules"],
    )
    if st.button("Run scientific validation", type="primary"):
        if val_source == "QM9 reference test set (full)":
            X_val = np.column_stack(
                [emb[f"true_{name}"].to_numpy(dtype=np.float64) for name in FEATURE_NAMES]
            )
        else:
            X_val = active.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64)

        if len(X_val) < 3:
            st.warning("At least 3 molecules are required for PCA. Using the QM9 reference test set instead.")
            X_val = np.column_stack(
                [emb[f"true_{name}"].to_numpy(dtype=np.float64) for name in FEATURE_NAMES]
            )

        # Fair baseline: same scaler fitted on the same training data as Q-PREP.
        X_scaled = model.scaler.transform(X_val)
        pca = PCA(n_components=3, random_state=42).fit(X_scaled)
        X_pca = model.scaler.inverse_transform(pca.inverse_transform(pca.transform(X_scaled)))

        _, X_qprep, q_error = qprep_engine(X_val)
        p_error = np.abs(
            (X_pca[:, L_IDX] - X_pca[:, H_IDX]) - X_pca[:, G_IDX]
        )

        st.session_state.validation = {
            "n": int(len(X_val)),
            "qprep_mean": float(np.mean(q_error)),
            "pca_mean": float(np.mean(p_error)),
            "qprep_p95": float(np.percentile(q_error, 95)),
            "pca_p95": float(np.percentile(p_error, 95)),
            "qprep_max": float(np.max(q_error)),
            "pca_max": float(np.max(p_error)),
            "threshold": threshold,
        }

    validation = st.session_state.validation
    if validation is not None:
        st.caption(f"Validated on {validation['n']:,} molecules")
        v1, v2, v3 = st.columns(3)
        v1.metric("Q-PREP mean violation", f"{validation['qprep_mean']:.6f} eV")
        v2.metric("PCA mean violation", f"{validation['pca_mean']:.6f} eV")
        v3.metric("Acceptance threshold", f"{threshold:.3f} eV")

        comp = pd.DataFrame(
            {
                "Method": ["Q-PREP (PINN)", "PCA (3 components)"],
                "Mean violation (eV)": [validation["qprep_mean"], validation["pca_mean"]],
            }
        )
        bar = px.bar(
            comp,
            x="Method",
            y="Mean violation (eV)",
            color="Method",
            text="Mean violation (eV)",
            color_discrete_map={"Q-PREP (PINN)": "#2ca02c", "PCA (3 components)": "#636efa"},
            labels={"Mean violation (eV)": "Mean |(LUMO − HOMO) − GAP| (eV)"},
            title="Physics Violation Score — lower is better",
        )
        bar.update_traces(texttemplate="%{text:.6f} eV", textposition="outside")
        bar.add_hline(
            y=threshold, line_dash="dash", line_color="#dc2626",
            annotation_text=f"Rejection threshold {threshold:.3f} eV",
        )
        bar.update_layout(height=430, showlegend=False, margin=dict(t=70, b=40))
        st.plotly_chart(bar, width="stretch")

        st.markdown(
            """
            **Physical Amnesia** is what happens when a compressed representation
            forgets the laws that made the original molecule physically meaningful.
            Q-PREP prevents it by putting the eV-space HOMO-LUMO residual directly
            into the training loss, so the decoder is penalised whenever it
            reconstructs an inconsistent gap, HOMO or LUMO.
            """
        )
        st.info(
            f"**Honest interpretation of this chart:** on real QM9 the observed means "
            f"are {validation['qprep_mean']:.6f} eV (Q-PREP) and "
            f"{validation['pca_mean']:.6f} eV (PCA). Both are far below the "
            f"{threshold:.3f} eV threshold. PCA's smaller value is expected: "
            "`gap = lumo − homo` is linear and PCA is a linear projector, so PCA "
            "preserves the identity by construction rather than by learning physics. "
            "Q-PREP's advantage is reconstruction fidelity (≈63% lower MSE on QM9) "
            "and robustness when molecules are edited or generated off-manifold."
        )
    else:
        st.info("Press **Run scientific validation** to compare the two compressors.")
