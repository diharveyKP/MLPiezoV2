"""
Quick FOS prediction using a trained ensemble.
Supports per-model config and model-rooted artifact naming.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CONFIG_DIR))

from config_utils import apply_synchronization, get_control_point_info_for_scripts
from runtime_context import get_runtime_config, resolve_ensemble_path


def load_ensemble(mode: str, model_dir: str | None = None, model_name: str | None = None):
    ensemble_path = resolve_ensemble_path(mode=mode, model_dir=model_dir, model_name=model_name)
    if not ensemble_path.exists():
        raise FileNotFoundError(f"Model not found: {ensemble_path}")
    with open(ensemble_path, "rb") as handle:
        return pickle.load(handle), ensemble_path


def predict_with_confidence(ensemble, control_points: list[float]):
    x_values = np.array(control_points, dtype=float).reshape(1, -1)
    fos_mean, fos_std, confidence = ensemble.predict_with_uncertainty(x_values)

    fos = float(fos_mean[0])
    unc = float(fos_std[0])
    conf = float(confidence[0])
    ci_lower = fos - 2 * unc
    ci_upper = fos + 2 * unc

    from scipy.stats import norm

    fail_prob = float(norm.cdf((1.0 - fos) / (unc + 1e-10)))

    return {
        "fos": fos,
        "uncertainty": unc,
        "confidence": conf,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "failure_probability": fail_prob,
    }


def print_prediction(result: dict, cp_values: list[float], cp_names: list[str]) -> None:
    print("\n" + "=" * 70)
    print("FOS PREDICTION")
    print("=" * 70)

    print("\nInput Configuration:")
    for name, value in zip(cp_names, cp_values):
        print(f"  {name:<30} {value:>8.1f} ft")

    print("\n" + "-" * 70)
    print("PREDICTION RESULTS")
    print("-" * 70)
    print(f"\n  FOS:          {result['fos']:.4f}")
    print(f"  Uncertainty:  +/-{result['uncertainty']:.4f}")
    print(f"  Confidence:   {result['confidence']*100:.1f}%")
    print(f"  95% CI:       [{result['ci_95_lower']:.3f}, {result['ci_95_upper']:.3f}]")
    print(f"  Failure Risk: {result['failure_probability']*100:.2f}%")

    print("\n" + "-" * 70)
    print("ASSESSMENT")
    print("-" * 70)
    fos = result["fos"]
    if fos < 1.0:
        print("  FAILURE (<1.0)")
        print("     Immediate action required.")
    elif fos < 1.3:
        print("  CRITICAL (1.0-1.3)")
        print("     Monitor closely and consider mitigation.")
    elif fos < 1.5:
        print("  MODERATE (1.3-1.5)")
        print("     Acceptable but requires monitoring.")
    else:
        print("  SAFE (>=1.5)")
        print("     Adequate factor of safety.")

    if result["confidence"] > 0.95:
        print("\n  HIGH CONFIDENCE - Prediction is reliable")
    elif result["confidence"] > 0.85:
        print("\n  MEDIUM CONFIDENCE - Consider validation")
    else:
        print("\n  LOW CONFIDENCE - Recommend GeoStudio verification")

    print("\n" + "=" * 70)


def interactive_mode(ensemble, cp_info: dict) -> None:
    print("\n" + "=" * 70)
    print("INTERACTIVE PREDICTION MODE")
    print("=" * 70)
    print("\nEnter control point elevations (or 'q' to quit)")

    while True:
        print("\n" + "-" * 70)
        cp_values = [0.0] * cp_info["count"]
        shown_indices = set()

        for group in cp_info["sync_groups"]:
            if group[0] in shown_indices:
                continue

            idx = group[0]
            name = " = ".join(cp_info["names"][i] for i in group)
            vmin, vmax = cp_info["bounds"][idx]

            while True:
                user_input = input(f"{name} [{vmin:.0f}-{vmax:.0f}] ft: ")
                if user_input.lower() == "q":
                    print("\nExiting...")
                    return
                try:
                    value = float(user_input)
                    if vmin <= value <= vmax:
                        for group_idx in group:
                            cp_values[group_idx] = value
                            shown_indices.add(group_idx)
                        break
                    print(f"  Value must be between {vmin:.0f} and {vmax:.0f}")
                except ValueError:
                    print("  Invalid number, try again")

        for idx, name in enumerate(cp_info["names"]):
            if idx in shown_indices:
                continue
            vmin, vmax = cp_info["bounds"][idx]
            while True:
                user_input = input(f"{name} [{vmin:.0f}-{vmax:.0f}] ft: ")
                if user_input.lower() == "q":
                    print("\nExiting...")
                    return
                try:
                    value = float(user_input)
                    if vmin <= value <= vmax:
                        cp_values[idx] = value
                        break
                    print(f"  Value must be between {vmin:.0f} and {vmax:.0f}")
                except ValueError:
                    print("  Invalid number, try again")

        result = predict_with_confidence(ensemble, cp_values)
        print_prediction(result, cp_values, cp_info["names"])

        again = input("\nAnother prediction? (y/n): ").lower()
        if again != "y":
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick FOS prediction")
    parser.add_argument("--mode", default="shallow", choices=["shallow", "deep"])
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--cp", nargs="+", type=float, help="Explicit control point values")
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("QUICK FOS PREDICTION")
    print("=" * 70)

    try:
        control_points_cfg, _, _, config_file = get_runtime_config(args.config_path)
        cp_info = get_control_point_info_for_scripts(control_points_cfg)

        ensemble, ensemble_path = load_ensemble(args.mode, args.model_dir, args.model_name)
        print(f"Loaded model: {ensemble_path}")
        if config_file is not None:
            print(f"Loaded config: {config_file}")
        print(f"Control points: {cp_info['count']}")

        if args.interactive:
            interactive_mode(ensemble, cp_info)
            return

        if args.cp:
            if len(args.cp) != cp_info["count"]:
                raise ValueError(f"Expected {cp_info['count']} control points, got {len(args.cp)}")
            cp_values = apply_synchronization(np.array(args.cp, dtype=float).reshape(1, -1), cp_info["sync_groups"])[0].tolist()
        else:
            cp_values = cp_info["y_baselines"].tolist()

        result = predict_with_confidence(ensemble, cp_values)
        print_prediction(result, cp_values, cp_info["names"])

        if not args.interactive and not args.cp:
            print("\nTip: use --interactive for custom predictions.")
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
