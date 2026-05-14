"""
Create summary plots from a completed hybrid overnight run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _load_status(status_path: Path) -> dict:
    return json.loads(status_path.read_text(encoding="utf-8"))


def _stages_dataframe(status: dict) -> pd.DataFrame:
    rows = []
    started_at = _parse_time(status["started_at"])
    for stage in status.get("stages", []):
        finished_at = _parse_time(stage["finished_at"])
        stats = stage.get("dataset_stats", {})
        rows.append(
            {
                "stage": stage["stage"],
                "finished_at": finished_at,
                "hours_from_start": (finished_at - started_at).total_seconds() / 3600.0,
                "return_code": stage["return_code"],
                "rows": stats.get("rows", 0),
                "successful_rows": stats.get("successful_rows", 0),
                "failed_rows": stats.get("failed_rows", 0),
                "stage_type": stage["stage"].split("_cycle_")[0] if "_cycle_" in stage["stage"] else stage["stage"],
            }
        )
    return pd.DataFrame(rows)


def create_plot(status_path: Path, output_path: Path) -> Path:
    status = _load_status(status_path)
    df = _stages_dataframe(status)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Hybrid Overnight Run Summary", fontsize=20, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df["hours_from_start"], df["rows"], color="#1f77b4", linewidth=2.5)
    ax.set_title("Dataset Growth", fontweight="bold")
    ax.set_xlabel("Hours from start")
    ax.set_ylabel("Rows")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    training_points = df[df["stage_type"] == "training"]
    active_points = df[df["stage_type"] == "active_learning"]
    explore_points = df[df["stage_type"] == "exploration_dataset_generation"]
    initial_points = df[df["stage_type"] == "initial_dataset_generation"]
    recovery_points = df[df["stage_type"] == "recovery_dataset_generation"]

    if len(initial_points):
        ax.scatter(initial_points["hours_from_start"], initial_points["rows"], label="Initial dataset", s=70, color="#2ca02c")
    if len(training_points):
        ax.scatter(training_points["hours_from_start"], training_points["rows"], label="Training", s=50, color="#1f77b4")
    if len(active_points):
        ax.scatter(active_points["hours_from_start"], active_points["rows"], label="Active learning", s=50, color="#d62728")
    if len(explore_points):
        ax.scatter(explore_points["hours_from_start"], explore_points["rows"], label="Exploration", s=50, color="#ff7f0e")
    if len(recovery_points):
        ax.scatter(recovery_points["hours_from_start"], recovery_points["rows"], label="Recovery", s=60, color="#9467bd")
    ax.set_title("Stage Timeline", fontweight="bold")
    ax.set_xlabel("Hours from start")
    ax.set_ylabel("Dataset rows at stage end")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    counts = df["stage_type"].value_counts().sort_values(ascending=False)
    ax.bar(counts.index, counts.values, color=["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"][: len(counts)])
    ax.set_title("Completed Stage Counts", fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.axis("off")
    final_stats = status.get("final_dataset_stats", {})
    summary_text = (
        f"Started: {status.get('started_at')}\n"
        f"Finished: {status.get('finished_at', '-')}\n"
        f"Deadline: {status.get('deadline')}\n\n"
        f"Cycles completed: {status.get('cycles_completed', 0)}\n"
        f"Rows: {final_stats.get('rows', 0)}\n"
        f"Successful: {final_stats.get('successful_rows', 0)}\n"
        f"Failed: {final_stats.get('failed_rows', 0)}\n\n"
        f"Mode: {status.get('mode')}\n"
        f"Model root: {status.get('model_root_name')}\n"
    )
    ax.text(
        0.02,
        0.98,
        summary_text,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    status_path = Path(args.status_path).resolve()
    output_path = Path(args.output_path).resolve() if args.output_path else status_path.parent / "HYBRID_SUMMARY.png"

    created = create_plot(status_path, output_path)
    print(f"Saved: {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
