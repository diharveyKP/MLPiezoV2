"""
Active Learning - GENERALIZED with Physics Validation
Intelligently select next samples, auto-adapts to config.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CONFIG_DIR))

from artifact_paths import build_ensemble_filename, resolve_model_output_dir, slugify
from config_utils import apply_synchronization, get_control_point_info_for_scripts
from geostudio_interface import GeoStudioInterface, PhreaticSurfaceData
from phreatic_generator import PhreaticSurfaceGenerator

try:
    from config import CONTROL_POINTS, DATASET, GEOSTUDIO
except ImportError as e:
    print(f"Cannot load config: {e}")
    sys.exit(1)


def load_runtime_config(config_path: str | None) -> None:
    """Load config either from the default module or an explicit file path."""
    global CONTROL_POINTS, DATASET, GEOSTUDIO

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
    GEOSTUDIO = module.GEOSTUDIO
    DATASET = module.DATASET
    print(f"Loaded config: {config_file}")


def load_ensemble(model_dir: str, mode: str, model_root_name: str | None = None):
    """Load ensemble using model-rooted artifact naming."""
    root_name = slugify(model_root_name) if model_root_name else slugify(Path(model_dir).name)
    model_file = resolve_model_output_dir(model_dir, root_name) / build_ensemble_filename(root_name, mode)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    with open(model_file, "rb") as f:
        return pickle.load(f)


def generate_candidates(n_candidates, bounds, n_features, sync_groups, seed=42):
    """Generate candidates using LHS."""
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=n_features, seed=seed)
    unit = sampler.random(n=n_candidates)

    candidates = np.zeros_like(unit)
    for i, (vmin, vmax) in enumerate(bounds):
        candidates[:, i] = vmin + unit[:, i] * (vmax - vmin)

    candidates = apply_synchronization(candidates, sync_groups)

    return np.round(candidates, 2)


def validate_flow_direction(control_y: np.ndarray, strict: bool = True) -> tuple[bool, str]:
    if len(control_y) < 2:
        return True, "OK"

    if strict:
        for i in range(1, len(control_y)):
            if control_y[i] > control_y[i - 1] + 0.01:
                return False, f"CP{i+1} ({control_y[i]:.1f}) > CP{i} ({control_y[i-1]:.1f})"
        return True, "OK"

    if control_y[-1] > control_y[0]:
        return False, f"Downstream ({control_y[-1]:.1f}) > Upstream ({control_y[0]:.1f})"
    return True, "OK"


def check_hydraulic_gradient(control_x: np.ndarray, control_y: np.ndarray, max_gradient: float = 2.0) -> tuple[bool, str]:
    for i in range(len(control_x) - 1):
        dx = control_x[i + 1] - control_x[i]
        dy = abs(control_y[i + 1] - control_y[i])

        if dx > 0:
            gradient = dy / dx
            if gradient > max_gradient:
                return False, f"Gradient {gradient:.2f} > {max_gradient} between CP{i+1} and CP{i+2}"

    return True, "OK"


def validate_candidate(control_y: np.ndarray, control_x: np.ndarray, max_gradient: float = 2.0) -> tuple[bool, str]:
    valid, reason = validate_flow_direction(control_y, strict=True)
    if not valid:
        return False, f"Flow direction: {reason}"

    valid, reason = check_hydraulic_gradient(control_x, control_y, max_gradient)
    if not valid:
        return False, f"Gradient: {reason}"

    return True, "OK"


def filter_valid_candidates(candidates: np.ndarray, cp_info: dict, max_gradient: float) -> np.ndarray:
    valid = []
    for candidate in candidates:
        is_valid, _ = validate_candidate(candidate, cp_info["x_coords"], max_gradient)
        if is_valid:
            valid.append(candidate)
    if not valid:
        raise ValueError("No valid candidates passed physics validation")
    return np.array(valid)


def select_most_uncertain(candidates, ensemble, n_select):
    """Select highest uncertainty samples."""
    print(f"\nEvaluating {len(candidates)} candidates...")

    fos_mean, fos_std, confidence = ensemble.predict_with_uncertainty(candidates)

    fos_std = np.atleast_1d(np.array(fos_std))
    fos_mean = np.atleast_1d(np.array(fos_mean))
    confidence = np.atleast_1d(np.array(confidence))

    print(f"  Uncertainty: [{fos_std.min():.4f}, {fos_std.max():.4f}]")
    print(f"  Mean uncertainty: {fos_std.mean():.4f}")

    n_select = min(n_select, len(candidates))
    top_idx = np.argsort(fos_std)[-n_select:][::-1]

    selected = candidates[top_idx]
    selected_unc = fos_std[top_idx]
    selected_fos = fos_mean[top_idx]

    print(f"\nSelected {n_select} samples")
    print(f"  Uncertainty: [{selected_unc.min():.4f}, {selected_unc.max():.4f}]")
    print(f"  Predicted FOS: [{selected_fos.min():.3f}, {selected_fos.max():.3f}]")

    return selected, top_idx, fos_std


def visualize(candidates, uncertainties, selected_idx, mode, output_dir, headless=False):
    """Visualize selection."""
    import matplotlib

    if headless:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Active Learning - {mode.upper()}", fontsize=20, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(uncertainties, bins=40, alpha=0.7, edgecolor="black", color="steelblue")
    threshold = uncertainties[selected_idx].min()
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2.5, label="Threshold")
    ax.set_xlabel("Uncertainty", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title("Uncertainty Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    y_axis_index = 2 if candidates.shape[1] > 2 else 1
    scatter = ax.scatter(
        candidates[:, 0],
        candidates[:, y_axis_index],
        c=uncertainties,
        cmap="viridis",
        s=30,
        alpha=0.5,
    )
    ax.scatter(
        candidates[selected_idx, 0],
        candidates[selected_idx, y_axis_index],
        s=200,
        c="red",
        marker="*",
        edgecolors="black",
        linewidth=1.5,
        label="Selected",
    )
    ax.set_xlabel("CP1 (ft)", fontweight="bold")
    ax.set_ylabel(f"CP{y_axis_index + 1} (ft)", fontweight="bold")
    ax.set_title("Parameter Space", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Uncertainty")

    ax = axes[1, 0]
    sorted_unc = np.sort(uncertainties)[::-1]
    ax.plot(sorted_unc, linewidth=2, color="navy")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=2)
    ax.fill_between(range(len(selected_idx)), 0, sorted_unc[: len(selected_idx)], alpha=0.3, color="red")
    ax.set_xlabel("Rank", fontweight="bold")
    ax.set_ylabel("Uncertainty", fontweight="bold")
    ax.set_title("Selection Curve", fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"SUMMARY\n"
        f"{'='*30}\n\n"
        f"Candidates: {len(candidates)}\n"
        f"Selected: {len(selected_idx)}\n\n"
        f"Mean unc: {uncertainties[selected_idx].mean():.4f}\n"
        f"Max unc: {uncertainties[selected_idx].max():.4f}\n\n"
        f"These samples improve\n"
        f"model in uncertain regions",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8, pad=1.5),
        family="monospace",
    )

    plt.tight_layout()

    out = output_dir / f"ACTIVE_LEARNING_{mode.upper()}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out.name}")

    if not headless:
        plt.show()
    plt.close(fig)


def next_sample_id(dataset_path: Path) -> int:
    """Compute the next available sample id."""
    if not dataset_path.exists():
        return 1

    df = pd.read_csv(dataset_path)
    if len(df) == 0 or "sample_id" not in df.columns:
        return 1

    return int(pd.to_numeric(df["sample_id"], errors="coerce").fillna(0).max()) + 1


def run_geostudio(selected, geostudio, phreatic_gen, mode, starting_sample_id):
    """Run GeoStudio for selected samples."""
    results = []

    print(f"\nRunning GeoStudio for {len(selected)} samples...")
    print("=" * 70)

    for offset, cp in enumerate(selected):
        sample_id = starting_sample_id + offset
        print(f"\n[{offset + 1}/{len(selected)}] Sample {sample_id}")

        try:
            y_interp = phreatic_gen.generate(cp)
            x_interp = phreatic_gen.get_x_coordinates()

            n_interp = len(y_interp)
            phreatic_data = PhreaticSurfaceData(
                control_point_numbers=list(range(1, n_interp + 1)),
                control_point_x=x_interp.tolist(),
                control_point_y=cp.tolist(),
                interpolated_x=x_interp,
                interpolated_y=y_interp,
            )

            result = geostudio.run_analysis(phreatic_data, sample_id)

            row = {
                "sample_id": sample_id,
                "mode": mode,
                "timestamp": datetime.now().isoformat(),
                "phreatic_surface_y": json.dumps(y_interp.tolist()),
                "success": result.success,
                "fos": result.fos if result.success else None,
                "solve_time": result.solve_time,
                "error_message": result.error_message,
                "source": "active_learning",
            }

            for j, y_value in enumerate(cp, start=1):
                row[f"control_y_{j}"] = y_value

            results.append(row)

            if result.success:
                print(f"  Success: FOS = {result.fos:.4f}")
            else:
                print("  Failed")

        except Exception as e:
            print(f"  Error: {e}")

    return pd.DataFrame(results)


def append_to_dataset(new_df: pd.DataFrame, dataset_path: Path):
    """Append active-learning results to dataset CSV."""
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        df_old = pd.read_csv(dataset_path)
        backup = dataset_path.with_suffix(".csv.backup_AL")
        df_old.to_csv(backup, index=False)
        df_new = pd.concat([df_old, new_df], ignore_index=True)
        before = len(df_old)
    else:
        df_new = new_df.copy()
        before = 0

    df_new.to_csv(dataset_path, index=False)

    print(f"\nUpdated dataset: {before} -> {len(df_new)} (+{len(new_df)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="shallow", choices=["shallow", "deep"])
    parser.add_argument("--n_candidates", type=int, default=1000)
    parser.add_argument("--n_select", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    load_runtime_config(args.config_path)

    print("=" * 70)
    print("ACTIVE LEARNING")
    print("=" * 70)

    try:
        cp_info = get_control_point_info_for_scripts(CONTROL_POINTS)
        print(f"Control points: {cp_info['count']}")

        ensemble = load_ensemble(args.model_dir, args.mode, args.model_name)

        print(f"\nGenerating {args.n_candidates} candidates...")
        candidates = generate_candidates(
            args.n_candidates,
            cp_info["bounds"],
            cp_info["count"],
            cp_info["sync_groups"],
            seed=args.seed,
        )
        candidates = filter_valid_candidates(
            candidates,
            cp_info,
            CONTROL_POINTS["constraints"]["max_gradient"],
        )
        print(f"Valid candidates after physics filter: {len(candidates)}")

        print(f"\nSelecting {args.n_select} most uncertain...")
        selected, idx, unc = select_most_uncertain(candidates, ensemble, args.n_select)

        output_dir = Path(DATASET["output_dir"])
        if not args.no_plot:
            visualize(candidates, unc, idx, args.mode, output_dir, headless=args.headless)

        if args.dry_run:
            print("\nDRY RUN - No GeoStudio")
            return

        print("\nSetting up GeoStudio...")
        template = GEOSTUDIO["templates"][args.mode]
        geocmd = Path(GEOSTUDIO["geocmd_location"]) / "geocmd.exe"
        geo = GeoStudioInterface(template, str(geocmd), GEOSTUDIO["fos_column_name"])

        phreatic = PhreaticSurfaceGenerator(
            control_point_x=cp_info["x_coords"].tolist(),
            n_total_points=cp_info["interpolation"]["n_total_points"],
            interpolation_method=cp_info["interpolation"]["method"],
            enforce_flow_direction=CONTROL_POINTS["constraints"]["enforce_flow_direction"],
        )

        dataset_path = Path(args.dataset_path) if args.dataset_path else Path(DATASET["output_dir"]) / f"{args.mode}.csv"
        starting_sample_id = next_sample_id(dataset_path)
        new_data = run_geostudio(selected, geo, phreatic, args.mode, starting_sample_id)

        if len(new_data) > 0:
            append_to_dataset(new_data, dataset_path)
            print("\nActive learning complete. Retrain next.")
        else:
            print("\nNo successful active-learning samples were added.")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
