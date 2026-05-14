"""
Sensitivity analysis for a trained FOS ensemble.
Produces a readable figure plus ranked CSV/JSON outputs.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CONFIG_DIR))

from artifact_paths import build_ensemble_filename, resolve_model_output_dir, slugify
from config_utils import get_control_point_info_for_scripts, print_control_point_summary

try:
    from config import CONTROL_POINTS, DATASET
except ImportError as exc:
    print(f"Cannot load config: {exc}")
    sys.exit(1)


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 120,
})


def load_runtime_config(config_path: str | None) -> None:
    global CONTROL_POINTS, DATASET

    if not config_path:
        print("Loaded config: configs/config.py")
        return

    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")

    spec = importlib.util.spec_from_file_location("runtime_model_config", config_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {config_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    CONTROL_POINTS = module.CONTROL_POINTS
    DATASET = module.DATASET
    print(f"Loaded config: {config_file}")


def load_ensemble(model_dir: str | Path, mode: str, model_root_name: str | None = None):
    root_name = slugify(model_root_name) if model_root_name else slugify(Path(model_dir).name)
    model_file = resolve_model_output_dir(model_dir, root_name) / build_ensemble_filename(root_name, mode)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    with open(model_file, "rb") as handle:
        return pickle.load(handle)


def load_dataset(dataset_dir: str | Path, mode: str) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    for name in [f"{mode}.csv", f"results_{mode}.csv", f"results_{mode}"]:
        candidate = dataset_dir / name
        if candidate.exists():
            df = pd.read_csv(candidate)
            return df[df["success"] == True].copy()
    raise FileNotFoundError(f"No dataset found for mode '{mode}' in {dataset_dir}")


def get_unique_groups(cp_info: dict) -> list[list[int]]:
    grouped = []
    seen = set()
    for group in cp_info["sync_groups"]:
        group_key = tuple(group)
        if group_key and group[0] not in seen:
            grouped.append(group)
            seen.update(group)
    for idx in range(cp_info["count"]):
        if idx not in seen:
            grouped.append([idx])
    return grouped


def group_label(cp_info: dict, group: list[int]) -> str:
    names = [cp_info["names"][idx] for idx in group]
    return " = ".join(names)


def compute_local_sensitivities(ensemble, baseline: np.ndarray, groups: list[list[int]]) -> dict[str, float]:
    delta = 1.0
    fos_base = float(ensemble.predict(baseline.reshape(1, -1))[0])
    sensitivities = {}

    for group in groups:
        x_up = baseline.copy()
        x_up[group] = baseline[group[0]] + delta
        fos_up = float(ensemble.predict(x_up.reshape(1, -1))[0])
        sensitivities[group_label(cp_info_global, group)] = (fos_up - fos_base) / delta

    return sensitivities


def compute_dataset_associations(df: pd.DataFrame, cp_info: dict, groups: list[list[int]]) -> dict[str, float]:
    associations = {}
    for group in groups:
        column = f"control_y_{group[0] + 1}"
        label = group_label(cp_info, group)
        corr = df[column].corr(df["fos"], method="spearman")
        associations[label] = float(corr) if pd.notna(corr) else 0.0
    return associations


def compute_partial_dependence(ensemble, baseline: np.ndarray, cp_info: dict, groups: list[list[int]]) -> list[dict]:
    panels = []
    for group in groups:
        idx = group[0]
        vmin, vmax = cp_info["bounds"][idx]
        values = np.linspace(vmin, vmax, 80)
        means = []
        stds = []

        for value in values:
            x_case = baseline.copy()
            x_case[group] = value
            pred_mean, pred_std, _ = ensemble.predict_with_uncertainty(x_case.reshape(1, -1))
            means.append(float(pred_mean[0]))
            stds.append(float(pred_std[0]))

        panels.append({
            "label": group_label(cp_info, group),
            "x": values,
            "mean": np.array(means),
            "std": np.array(stds),
            "baseline_x": float(baseline[idx]),
        })
    return panels


def save_rankings(output_dir: Path, mode: str, sensitivities: dict[str, float], associations: dict[str, float]) -> None:
    rows = []
    ordered = sorted(sensitivities.items(), key=lambda item: abs(item[1]), reverse=True)
    for rank, (feature, local_sensitivity) in enumerate(ordered, start=1):
        rows.append({
            "feature": feature,
            "local_sensitivity_fos_per_ft": local_sensitivity,
            "local_direction": "down" if local_sensitivity < 0 else "up",
            "dataset_spearman_correlation": associations.get(feature, 0.0),
            "rank": rank,
        })

    csv_path = output_dir / f"SENSITIVITY_{mode.upper()}_RANKING.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / f"SENSITIVITY_{mode.upper()}_RANKING.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Saved: {csv_path.name}")
    print(f"Saved: {json_path.name}")


def create_plots(
    ensemble,
    cp_info: dict,
    df: pd.DataFrame,
    mode: str,
    output_dir: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    baseline = cp_info["y_baselines"]
    groups = get_unique_groups(cp_info)
    sensitivities = compute_local_sensitivities(ensemble, baseline, groups)
    associations = compute_dataset_associations(df, cp_info, groups)
    pd_panels = compute_partial_dependence(ensemble, baseline, cp_info, groups)

    ordered = sorted(sensitivities.items(), key=lambda item: abs(item[1]), reverse=True)
    feature_order = [name for name, _ in ordered]
    local_values = [value for _, value in ordered]
    assoc_values = [associations.get(name, 0.0) for name in feature_order]

    fig = plt.figure(figsize=(18, 12))
    outer = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.0, 1.1], hspace=0.42, wspace=0.35)
    fig.suptitle(f"Sensitivity Analysis - {mode.upper()}", fontsize=20, fontweight="bold", y=0.985)

    ax_local = fig.add_subplot(outer[0, :2])
    colors = ["#d95f5f" if value < 0 else "#4daf7c" for value in local_values]
    bars = ax_local.barh(feature_order, local_values, color=colors, edgecolor="black", linewidth=0.8)
    ax_local.axvline(0.0, color="black", linewidth=1.2)
    ax_local.invert_yaxis()
    ax_local.set_title("Local Sensitivity at Baseline", fontweight="bold")
    ax_local.set_xlabel("Delta FOS per 1 ft change")
    ax_local.grid(True, axis="x", alpha=0.25)
    x_extent = max(abs(value) for value in local_values) if local_values else 0.001
    ax_local.set_xlim(-x_extent * 1.18, x_extent * 1.18)
    for bar, value in zip(bars, local_values):
        offset = x_extent * 0.03
        x_pos = value - offset if value < 0 else value + offset
        ha = "right" if value < 0 else "left"
        ax_local.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{value:+.4f}", va="center", ha=ha, fontsize=9)

    ax_assoc = fig.add_subplot(outer[0, 2:])
    assoc_bars = ax_assoc.barh(feature_order, assoc_values, color="#5b9bd5", edgecolor="black", linewidth=0.8)
    ax_assoc.axvline(0.0, color="black", linewidth=1.2)
    ax_assoc.invert_yaxis()
    ax_assoc.set_title("Dataset Association (Spearman)", fontweight="bold")
    ax_assoc.set_xlabel("Correlation with FOS")
    ax_assoc.grid(True, axis="x", alpha=0.25)
    assoc_extent = max(abs(value) for value in assoc_values) if assoc_values else 0.1
    ax_assoc.set_xlim(-max(0.1, assoc_extent * 1.18), max(0.1, assoc_extent * 1.18))
    for bar, value in zip(assoc_bars, assoc_values):
        offset = max(0.01, assoc_extent * 0.05)
        x_pos = value - offset if value < 0 else value + offset
        ha = "right" if value < 0 else "left"
        ax_assoc.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}", va="center", ha=ha, fontsize=9)

    pd_grid = outer[1:, :].subgridspec(2, 4, hspace=0.42, wspace=0.30)
    for panel_idx, panel in enumerate(pd_panels):
        ax = fig.add_subplot(pd_grid[panel_idx // 4, panel_idx % 4])
        ax.plot(panel["x"], panel["mean"], color="#1f77b4", linewidth=2.2)
        ax.fill_between(panel["x"], panel["mean"] - panel["std"], panel["mean"] + panel["std"],
                        color="#1f77b4", alpha=0.18)
        ax.axhline(1.0, color="#cc0000", linestyle="--", linewidth=1.3, alpha=0.75)
        ax.axhline(1.25, color="#f4a300", linestyle=":", linewidth=1.3, alpha=0.75)
        ax.axhline(1.3, color="#2ca02c", linestyle="--", linewidth=1.3, alpha=0.75)
        baseline_fos = float(ensemble.predict(baseline.reshape(1, -1))[0])
        ax.scatter([panel["baseline_x"]], [baseline_fos], color="#c00000", marker="*", s=120,
                   edgecolors="black", linewidth=0.8, zorder=5)
        ax.set_title(panel["label"], fontweight="bold")
        ax.set_xlabel("Elevation (ft)")
        if panel_idx % 4 == 0:
            ax.set_ylabel("Predicted FOS")
        ax.grid(True, alpha=0.22)

    for panel_idx in range(len(pd_panels), 8):
        ax = fig.add_subplot(pd_grid[panel_idx // 4, panel_idx % 4])
        ax.axis("off")

    note = (
        f"Rows analyzed: {len(df):,}\n"
        f"FOS range: {df['fos'].min():.4f} to {df['fos'].max():.4f}\n"
        f"Most influential: {feature_order[0]} ({local_values[0]:+.4f} FOS/ft)\n"
        f"Interpretation: negative values mean higher water level lowers stability."
    )
    fig.text(0.012, 0.012, note, ha="left", va="bottom", fontsize=10,
             bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"})

    out = output_dir / f"SENSITIVITY_{mode.upper()}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    print(f"Saved: {out.name}")
    plt.close(fig)

    save_rankings(output_dir, mode, sensitivities, associations)
    return sensitivities, associations


def print_report(sensitivities: dict[str, float], associations: dict[str, float]) -> None:
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)

    sorted_sens = sorted(sensitivities.items(), key=lambda item: abs(item[1]), reverse=True)
    for name, value in sorted_sens:
        direction = "decreases" if value < 0 else "increases"
        assoc = associations.get(name, 0.0)
        print(
            f"  {name:<18} local={value:+.4f} FOS/ft  "
            f"dataset_spearman={assoc:+.3f}  higher water {direction} stability"
        )

    if sorted_sens:
        strongest_name, strongest_value = sorted_sens[0]
        print("-" * 70)
        print(f"Most influential control point/group: {strongest_name} ({strongest_value:+.4f} FOS per 1 ft)")


cp_info_global: dict = {}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="shallow", choices=["shallow", "deep"])
    parser.add_argument("--models", default="models")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)

    try:
        load_runtime_config(args.config_path)
        ensemble = load_ensemble(args.models, args.mode, args.model_name)
        dataset_dir = Path(args.dataset) if args.dataset else Path(DATASET["output_dir"])
        df = load_dataset(dataset_dir, args.mode)

        global cp_info_global
        cp_info_global = get_control_point_info_for_scripts(CONTROL_POINTS)

        print(f"\nControl points: {cp_info_global['count']}")
        print(f"Dataset rows used: {len(df):,}")
        print_control_point_summary(CONTROL_POINTS)

        sensitivities, associations = create_plots(
            ensemble=ensemble,
            cp_info=cp_info_global,
            df=df,
            mode=args.mode,
            output_dir=dataset_dir,
        )
        print_report(sensitivities, associations)
        print("\nComplete!")
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
