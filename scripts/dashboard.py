"""
Streamlit dashboard for forward prediction, dataset review, and inverse search.

Set optional environment variables before launching:
- MLPIEZOV2_CONFIG_PATH
- MLPIEZOV2_MODEL_DIR
- MLPIEZOV2_MODEL_NAME
- MLPIEZOV2_DATASET_DIR
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import differential_evolution
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CONFIG_DIR))

from config_utils import apply_synchronization, get_control_point_info_for_scripts
from runtime_context import get_runtime_config, resolve_dataset_dir, resolve_ensemble_path, resolve_model_dir, resolve_model_name


st.set_page_config(page_title="MLPiezoV2 Dashboard", page_icon="O", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
    .main { background-color: #ffffff; }
    .stMetric {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        border: 2px solid #000000;
    }
    .stMetric label { color: #000000 !important; font-weight: 700 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #000000 !important; font-size: 26px !important; font-weight: 800 !important; }
    h1, h2, h3 { color: #000000 !important; font-weight: 700 !important; }
    .stButton > button {
        background-color: #000000;
        color: #ffffff;
        font-weight: 700;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stButton > button:hover { background-color: #333333; }
    [data-testid="stSidebar"] { background-color: #f8f8f8; border-right: 3px solid #000; }
    .success-box, .warning-box, .danger-box {
        background-color: #f8f8f8;
        border: 3px solid #000;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


CONTROL_POINTS_CFG, DATASET_CFG, _, CONFIG_FILE = get_runtime_config(None)
CP_INFO = get_control_point_info_for_scripts(CONTROL_POINTS_CFG)
DEFAULT_MODEL_DIR = str(resolve_model_dir(None))
DEFAULT_MODEL_NAME = resolve_model_name(None, DEFAULT_MODEL_DIR)
DEFAULT_DATASET_DIR = str(resolve_dataset_dir(None, DATASET_CFG))


@st.cache_resource
def load_ensemble_cached(mode: str, model_dir: str, model_name: str):
    model_path = resolve_ensemble_path(mode=mode, model_dir=model_dir, model_name=model_name)
    if not model_path.exists():
        return None, model_path
    with open(model_path, "rb") as handle:
        return pickle.load(handle), model_path


@st.cache_data
def load_dataset_cached(dataset_dir: str, mode: str):
    root = Path(dataset_dir)
    for name in [f"{mode}.csv", f"results_{mode}.csv", f"results_{mode}"]:
        candidate = root / name
        if candidate.exists():
            return pd.read_csv(candidate)
    return None


def create_gauge(fos: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fos,
            number={"font": {"size": 32, "color": "#000"}},
            gauge={
                "axis": {"range": [0.8, 2.0], "tickwidth": 2, "tickcolor": "#000", "tickfont": {"size": 10}},
                "bar": {"color": "#000", "thickness": 0.75},
                "bgcolor": "#fff",
                "borderwidth": 3,
                "bordercolor": "#000",
                "steps": [
                    {"range": [0.8, 1.0], "color": "#ffcccc"},
                    {"range": [1.0, 1.3], "color": "#ffe6cc"},
                    {"range": [1.3, 2.0], "color": "#ccffcc"},
                ],
                "threshold": {"line": {"color": "#cc0000", "width": 4}, "value": 1.0},
            },
            title={"text": "<b>FOS</b>", "font": {"size": 14, "color": "#000"}},
        )
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="#fff")
    return fig


def page_home(ensemble, cp_info: dict) -> None:
    st.title("MLPiezoV2 Prediction")
    st.markdown("---")

    col_in, col_out = st.columns([1, 1.5])
    with col_in:
        st.markdown("### Control Points")
        cp_values = [0.0] * cp_info["count"]
        shown_indices = set()

        for group in cp_info["sync_groups"]:
            if group[0] in shown_indices:
                continue
            idx = group[0]
            vmin, vmax = cp_info["bounds"][idx]
            label = " & ".join(cp_info["names"][i] for i in group)
            value = st.slider(label, min_value=float(vmin), max_value=float(vmax), value=float(vmax), step=0.5, key=f"group_{idx}")
            for group_idx in group:
                cp_values[group_idx] = value
                shown_indices.add(group_idx)
            st.caption(f"Synchronized: {' = '.join([f'CP{i+1}' for i in group])} = {value:.1f} ft")

        for idx in range(cp_info["count"]):
            if idx in shown_indices:
                continue
            vmin, vmax = cp_info["bounds"][idx]
            cp_values[idx] = st.slider(cp_info["names"][idx], min_value=float(vmin), max_value=float(vmax), value=float(vmax), step=0.5, key=f"cp_{idx}")

        st.markdown("---")
        config_df = pd.DataFrame(
            {
                "Point": [f"CP{i+1}" for i in range(cp_info["count"])],
                "Name": cp_info["names"],
                "Elevation": [f"{value:.1f} ft" for value in cp_values],
            }
        )
        st.dataframe(config_df, hide_index=True, width="stretch")

    with col_out:
        st.markdown("### Prediction")
        x_values = np.array(cp_values, dtype=float).reshape(1, -1)
        fos_mean, fos_std, confidence = ensemble.predict_with_uncertainty(x_values)
        fos = float(fos_mean[0])
        unc = float(fos_std[0])
        conf = float(confidence[0])

        st.plotly_chart(create_gauge(fos), width="stretch")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("FOS", f"{fos:.4f}")
        m2.metric("Uncertainty", f"+/-{unc:.4f}")
        m3.metric("Confidence", f"{conf*100:.0f}%")
        risk = norm.cdf((1.0 - fos) / (unc + 1e-10)) * 100
        m4.metric("Risk", f"{risk:.1f}%")

        ci_l = fos - 2 * unc
        ci_u = fos + 2 * unc
        st.markdown(f"**95% CI:** [{ci_l:.3f}, {ci_u:.3f}]")

        if fos < 1.0:
            st.markdown("<div class='danger-box'><h3>FAILURE</h3></div>", unsafe_allow_html=True)
        elif fos < 1.3:
            st.markdown("<div class='warning-box'><h3>CRITICAL</h3></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='success-box'><h3>SAFE</h3></div>", unsafe_allow_html=True)


def page_dataset(df: pd.DataFrame | None) -> None:
    st.title("Dataset")
    st.markdown("---")
    if df is None:
        st.warning("No dataset found.")
        return

    df_ok = df[df["success"] == True].copy()
    fos = df_ok["fos"].to_numpy()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples", len(df))
    m2.metric("Successful", len(df_ok))
    m3.metric("Range", f"{fos.min():.3f}-{fos.max():.3f}")
    m4.metric("Mean", f"{fos.mean():.3f}")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=fos, nbinsx=30, marker=dict(color="steelblue", line=dict(color="#000", width=1.2))))
    fig.add_vline(x=fos.mean(), line_dash="dash", line_color="#000", line_width=3)
    fig.add_vline(x=1.0, line_dash="solid", line_color="#cc0000", line_width=3)
    fig.update_layout(title="<b>FOS Distribution</b>", xaxis_title="<b>FOS</b>", yaxis_title="<b>Count</b>", template="plotly_white", height=400)
    st.plotly_chart(fig, width="stretch")


def page_inverse(ensemble, cp_info: dict) -> None:
    st.title("Inverse Design")
    st.markdown("---")
    target = st.number_input("Target FOS", 0.9, 2.0, 1.3, 0.05)
    if st.button("Find Configuration", type="primary"):
        with st.spinner("Optimizing..."):
            def objective(x_values):
                x_sync = apply_synchronization(x_values.reshape(1, -1), cp_info["sync_groups"])[0]
                return abs(ensemble.predict(x_sync.reshape(1, -1))[0] - target)

            result = differential_evolution(objective, cp_info["bounds"], seed=42, maxiter=150)
            optimal = apply_synchronization(result.x.reshape(1, -1), cp_info["sync_groups"])[0]
            achieved = float(ensemble.predict(optimal.reshape(1, -1))[0])

        st.success("Solution found")
        m1, m2 = st.columns(2)
        m1.metric("Target", f"{target:.3f}")
        m2.metric("Achieved", f"{achieved:.3f}")
        result_df = pd.DataFrame({"Point": [f"CP{i+1}" for i in range(cp_info["count"])], "Name": cp_info["names"], "Optimal": [f"{value:.1f}" for value in optimal]})
        st.dataframe(result_df, hide_index=True, width="stretch")


def sidebar(cp_info: dict) -> tuple[str, str, str, str]:
    with st.sidebar:
        st.markdown("# MLPiezoV2")
        st.markdown("---")
        page = st.radio("Navigation", ["Home", "Dataset", "Inverse"])
        mode = st.selectbox("Mode", ["shallow", "deep"])
        model_dir = st.text_input("Model dir", value=DEFAULT_MODEL_DIR)
        model_name = st.text_input("Model name", value=DEFAULT_MODEL_NAME)
        dataset_dir = st.text_input("Dataset dir", value=DEFAULT_DATASET_DIR)
        st.markdown("---")
        st.caption(f"{cp_info['count']} control points")
        st.caption(f"{cp_info['interpolation']['n_total_points']} interpolated points")
        if CONFIG_FILE is not None:
            st.caption(f"Config: {CONFIG_FILE.name}")
        return page, mode, model_dir, model_name, dataset_dir


def main() -> None:
    page, mode, model_dir, model_name, dataset_dir = sidebar(CP_INFO)
    ensemble, model_path = load_ensemble_cached(mode, model_dir, model_name)
    dataset = load_dataset_cached(dataset_dir, mode)

    with st.sidebar:
        st.markdown("---")
        if ensemble is not None:
            st.success("Model ready")
            st.caption(str(model_path))
        else:
            st.error("No model found")

    if page == "Home":
        if ensemble is None:
            st.error("Train or point to a model first.")
        else:
            page_home(ensemble, CP_INFO)
    elif page == "Dataset":
        page_dataset(dataset)
    else:
        if ensemble is None:
            st.error("No model found.")
        else:
            page_inverse(ensemble, CP_INFO)

    st.markdown("---")
    st.caption("MLPiezoV2 | Clean ML workflow repo")


if __name__ == "__main__":
    main()
