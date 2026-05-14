"""
Inverse design for target FOS using a trained ensemble.
Supports per-model config and model-rooted artifact naming.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CONFIG_DIR))

from config_utils import apply_synchronization, get_control_point_info_for_scripts, print_control_point_summary
from runtime_context import get_runtime_config, resolve_dataset_dir, resolve_ensemble_path


def load_ensemble(mode: str, model_dir: str | None = None, model_name: str | None = None):
    ensemble_path = resolve_ensemble_path(mode=mode, model_dir=model_dir, model_name=model_name)
    if not ensemble_path.exists():
        raise FileNotFoundError(f"Model not found: {ensemble_path}")
    with open(ensemble_path, "rb") as handle:
        return pickle.load(handle), ensemble_path


def find_target_configuration(ensemble, target_fos: float, cp_info: dict, method: str = "global") -> dict:
    bounds = cp_info["bounds"]
    sync_groups = cp_info["sync_groups"]

    def objective(x_values):
        x_sync = apply_synchronization(x_values.reshape(1, -1), sync_groups)[0]
        pred = ensemble.predict(x_sync.reshape(1, -1))[0]
        return abs(pred - target_fos)

    print(f"\nSearching for FOS = {target_fos:.3f}...")
    print(f"  Method: {method}")

    if method == "global":
        result = differential_evolution(objective, bounds, seed=42, maxiter=300, popsize=15, atol=0.001)
    else:
        from scipy.optimize import minimize

        x0 = np.array([(bounds_i[0] + bounds_i[1]) / 2 for bounds_i in bounds], dtype=float)
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    optimal = apply_synchronization(result.x.reshape(1, -1), sync_groups)[0]
    achieved = float(ensemble.predict(optimal.reshape(1, -1))[0])
    _, unc, conf = ensemble.predict_with_uncertainty(optimal.reshape(1, -1))

    return {
        "optimal": optimal,
        "target": target_fos,
        "achieved": achieved,
        "error": abs(achieved - target_fos),
        "uncertainty": float(unc[0]),
        "confidence": float(conf[0]),
        "success": bool(result.success),
    }


def visualize_solution(result: dict, cp_info: dict, mode: str, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Inverse Design - {mode.upper()} - Target FOS = {result['target']:.3f}", fontsize=18, fontweight="bold")

    optimal = result["optimal"]
    bounds = cp_info["bounds"]

    ax = axes[0, 0]
    x_pos = np.arange(cp_info["count"])
    bars = ax.bar(x_pos, optimal, alpha=0.8, edgecolor="black", linewidth=1.2, color="#3498DB")
    for idx, (bounds_i, value) in enumerate(zip(bounds, optimal)):
        ymin, ymax = bounds_i
        ax.plot([idx, idx], [ymin, ymax], "k-", linewidth=2, alpha=0.3)
        ax.plot([idx], [ymin], "kv", markersize=7)
        ax.plot([idx], [ymax], "k^", markersize=7)
        ax.text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"CP{i+1}" for i in range(cp_info["count"])], rotation=0)
    ax.set_ylabel("Elevation (ft)", fontweight="bold")
    ax.set_title("Optimal Configuration", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.axis("off")
    status_color = "#27AE60" if result["error"] < 0.01 else ("#F39C12" if result["error"] < 0.05 else "#E74C3C")
    summary = (
        f"SOLUTION QUALITY\n{'='*35}\n\n"
        f"Target:      {result['target']:.4f}\n"
        f"Achieved:    {result['achieved']:.4f}\n"
        f"Error:       {result['error']:.4f}\n\n"
        f"Confidence:  {result['confidence']*100:.1f}%\n"
        f"Uncertainty: +/-{result['uncertainty']:.4f}\n"
    )
    ax.text(0.5, 0.5, summary, ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round", facecolor=status_color, alpha=0.25, pad=1.3),
            family="monospace", fontweight="bold")

    ax = axes[1, 0]
    current_max = np.array([bounds_i[1] for bounds_i in bounds], dtype=float)
    deltas = optimal - current_max
    colors = ["#27AE60" if delta < 0 else "#E74C3C" for delta in deltas]
    bars = ax.barh(range(cp_info["count"]), deltas, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)
    ax.axvline(0, color="black", linewidth=1.4)
    ax.set_yticks(range(cp_info["count"]))
    ax.set_yticklabels([f"CP{i+1}" for i in range(cp_info["count"])])
    ax.set_xlabel("Change from Max (ft)", fontweight="bold")
    ax.set_title("Required Adjustments", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    for bar, delta in zip(bars, deltas):
        x_pos_text = delta - 0.5 if delta < 0 else delta + 0.5
        ha = "right" if delta < 0 else "left"
        ax.text(x_pos_text, bar.get_y() + bar.get_height() / 2, f"{delta:+.1f}", va="center", ha=ha, fontsize=9, fontweight="bold")

    ax = axes[1, 1]
    ax.axis("off")
    recommendations = [f"Expected FOS: {result['achieved']:.3f}", f"Confidence: {result['confidence']*100:.0f}%"]
    for group in cp_info["sync_groups"]:
        delta = optimal[group[0]] - bounds[group[0]][1]
        if delta < -1:
            group_label = " & ".join(cp_info["names"][idx] for idx in group)
            recommendations.append(f"{group_label}: lower to <= {optimal[group[0]]:.1f} ft (Delta = {delta:.1f} ft)")
    ax.text(0.03, 0.97, "RECOMMENDATIONS\n" + "=" * 32 + "\n\n" + "\n".join(recommendations),
            ha="left", va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8, pad=1.2),
            family="monospace", transform=ax.transAxes)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"INVERSE_{mode.upper()}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_report(result: dict, cp_info: dict) -> None:
    print("\n" + "=" * 70)
    print("SOLUTION")
    print("=" * 70)
    print(f"  Target:      {result['target']:.4f}")
    print(f"  Achieved:    {result['achieved']:.4f}")
    print(f"  Error:       {result['error']:.4f}")
    print(f"  Confidence:  {result['confidence']*100:.1f}%")
    print(f"  Uncertainty: +/-{result['uncertainty']:.4f}")
    print("\nOptimal Control Points:")
    for name, value in zip(cp_info["names"], result["optimal"]):
        print(f"  {name:<30} {value:>8.2f} ft")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--mode", default="shallow", choices=["shallow", "deep"])
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--method", default="global", choices=["global", "local"])
    args = parser.parse_args()

    print("=" * 70)
    print("INVERSE DESIGN")
    print("=" * 70)
    print(f"Target FOS: {args.target:.3f}")

    try:
        control_points_cfg, dataset_cfg, _, config_file = get_runtime_config(args.config_path)
        cp_info = get_control_point_info_for_scripts(control_points_cfg)
        print_control_point_summary(control_points_cfg)

        ensemble, ensemble_path = load_ensemble(args.mode, args.model_dir, args.model_name)
        print(f"Loaded model: {ensemble_path}")
        if config_file is not None:
            print(f"Loaded config: {config_file}")

        result = find_target_configuration(ensemble, args.target, cp_info, args.method)
        print_report(result, cp_info)

        output_dir = resolve_dataset_dir(args.dataset_dir, dataset_cfg)
        image_path = visualize_solution(result, cp_info, args.mode, output_dir)
        print(f"Saved: {image_path}")
        print("\nComplete!")
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
