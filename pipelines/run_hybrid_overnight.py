#!/usr/bin/env python3
"""
Hybrid overnight runner:
dataset generation -> training -> active learning -> dataset generation -> repeat until time limit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


PIPELINE_DIR = Path(__file__).parent
PROJECT_ROOT = PIPELINE_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def load_runtime_config(config_path: str | Path):
    config_file = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("runtime_model_config", config_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {config_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_command(command: list[str], stage_name: str) -> int:
    print("\n" + "=" * 78)
    print(stage_name)
    print("=" * 78)
    print("Command:")
    print("  " + " ".join(f'"{part}"' if " " in part else part for part in command))
    print("=" * 78)

    process = subprocess.Popen(command)
    return process.wait()


def _dataset_stats(dataset_path: Path) -> dict:
    if not dataset_path.exists():
        return {
            "exists": False,
            "rows": 0,
            "successful_rows": 0,
            "failed_rows": 0,
        }

    df = pd.read_csv(dataset_path)
    success_count = int((df["success"] == True).sum()) if "success" in df.columns else 0
    return {
        "exists": True,
        "rows": int(len(df)),
        "successful_rows": success_count,
        "failed_rows": int(len(df) - success_count),
    }


def _write_status(status_path: Path, payload: dict) -> None:
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _start_stage(status: dict, status_path: Path, stage: str, command: list[str]) -> None:
    status["current_stage"] = stage
    status["current_stage_started_at"] = datetime.now().isoformat()
    status["current_command"] = command
    _write_status(status_path, status)


def _finish_stage(status: dict, status_path: Path) -> None:
    status["current_stage"] = None
    status["current_stage_started_at"] = None
    status["current_command"] = None
    _write_status(status_path, status)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the hybrid exploration + active-learning workflow until a time limit")
    parser.add_argument("--config-path", required=True, help="Path to model_config.py")
    parser.add_argument("--mode", default="shallow", choices=["shallow", "deep"])
    parser.add_argument("--hours", type=float, required=True, help="How long to run the hybrid loop")
    parser.add_argument("--initial-samples", type=int, default=100)
    parser.add_argument("--exploration-samples", type=int, default=25)
    parser.add_argument("--n-candidates", type=int, default=1000)
    parser.add_argument("--n-select", type=int, default=25)
    parser.add_argument("--train-min-samples", type=int, default=20)
    parser.add_argument("--headless", action="store_true", help="Do not open plots during active learning")
    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    config_module = load_runtime_config(config_path)

    workspace_root = config_path.parent
    model_root_name = workspace_root.name
    dataset_dir = Path(config_module.DATASET["output_dir"]).resolve()
    dataset_path = dataset_dir / f"{args.mode}.csv"
    trained_models_dir = (workspace_root / "trained_models").resolve()
    status_path = (workspace_root / "hybrid_run_status.json").resolve()

    python_exe = sys.executable
    deadline = datetime.now() + timedelta(hours=args.hours)

    status = {
        "started_at": datetime.now().isoformat(),
        "deadline": deadline.isoformat(),
        "config_path": str(config_path),
        "mode": args.mode,
        "workspace_root": str(workspace_root),
        "dataset_dir": str(dataset_dir),
        "dataset_path": str(dataset_path),
        "trained_models_dir": str(trained_models_dir),
        "model_root_name": model_root_name,
        "cycles_completed": 0,
        "current_stage": None,
        "current_stage_started_at": None,
        "current_command": None,
        "stages": [],
    }
    _write_status(status_path, status)

    def record_stage(stage: str, return_code: int) -> None:
        status["stages"].append(
            {
                "stage": stage,
                "finished_at": datetime.now().isoformat(),
                "return_code": return_code,
                "dataset_stats": _dataset_stats(dataset_path),
            }
        )
        _write_status(status_path, status)

    generate_command = [
        python_exe,
        str(PIPELINE_DIR / "pipeline_generate_dataset.py"),
        "--mode",
        args.mode,
        "--n_samples",
        str(args.initial_samples),
        "--config-path",
        str(config_path),
    ]
    _start_stage(status, status_path, "initial_dataset_generation", generate_command)
    rc = _run_command(generate_command, "INITIAL DATASET GENERATION")
    record_stage("initial_dataset_generation", rc)
    _finish_stage(status, status_path)
    if rc != 0:
        return rc

    cycle = 0
    while datetime.now() < deadline:
        cycle += 1
        status["cycles_completed"] = cycle
        _write_status(status_path, status)

        train_command = [
            python_exe,
            str(PIPELINE_DIR / "train_models.py"),
            "--dataset",
            str(dataset_dir),
            "--mode",
            args.mode,
            "--output",
            str(trained_models_dir),
            "--model-name",
            model_root_name,
            "--min_samples",
            str(args.train_min_samples),
        ]
        _start_stage(status, status_path, f"training_cycle_{cycle}", train_command)
        rc = _run_command(train_command, f"TRAINING CYCLE {cycle}")
        record_stage(f"training_cycle_{cycle}", rc)
        _finish_stage(status, status_path)
        if rc != 0:
            if datetime.now() >= deadline:
                break
            recovery_command = [
                python_exe,
                str(PIPELINE_DIR / "pipeline_generate_dataset.py"),
                "--mode",
                args.mode,
                "--n_samples",
                str(args.exploration_samples),
                "--config-path",
                str(config_path),
            ]
            _start_stage(status, status_path, f"recovery_dataset_generation_cycle_{cycle}", recovery_command)
            rc = _run_command(recovery_command, f"RECOVERY DATASET GENERATION CYCLE {cycle}")
            record_stage(f"recovery_dataset_generation_cycle_{cycle}", rc)
            _finish_stage(status, status_path)
            if rc != 0:
                return rc
            continue

        if datetime.now() >= deadline:
            break

        active_learning_command = [
            python_exe,
            str(SCRIPTS_DIR / "active_learning.py"),
            "--mode",
            args.mode,
            "--n_candidates",
            str(args.n_candidates),
            "--n_select",
            str(args.n_select),
            "--config-path",
            str(config_path),
            "--model-dir",
            str(trained_models_dir),
            "--model-name",
            model_root_name,
            "--dataset-path",
            str(dataset_path),
        ]
        if args.headless:
            active_learning_command.extend(["--headless", "--no-plot"])

        _start_stage(status, status_path, f"active_learning_cycle_{cycle}", active_learning_command)
        rc = _run_command(active_learning_command, f"ACTIVE LEARNING CYCLE {cycle}")
        record_stage(f"active_learning_cycle_{cycle}", rc)
        _finish_stage(status, status_path)
        if rc != 0:
            return rc

        if datetime.now() >= deadline:
            break

        explore_command = [
            python_exe,
            str(PIPELINE_DIR / "pipeline_generate_dataset.py"),
            "--mode",
            args.mode,
            "--n_samples",
            str(args.exploration_samples),
            "--config-path",
            str(config_path),
        ]
        _start_stage(status, status_path, f"exploration_dataset_generation_cycle_{cycle}", explore_command)
        rc = _run_command(explore_command, f"EXPLORATION DATASET GENERATION CYCLE {cycle}")
        record_stage(f"exploration_dataset_generation_cycle_{cycle}", rc)
        _finish_stage(status, status_path)
        if rc != 0:
            return rc

    status["finished_at"] = datetime.now().isoformat()
    status["final_dataset_stats"] = _dataset_stats(dataset_path)
    _write_status(status_path, status)

    print("\n" + "=" * 78)
    print("HYBRID RUN COMPLETE")
    print("=" * 78)
    print(f"Status file: {status_path}")
    print(f"Dataset stats: {status['final_dataset_stats']}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
