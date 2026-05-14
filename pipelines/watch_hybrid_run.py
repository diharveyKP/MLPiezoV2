#!/usr/bin/env python3
"""
CLI dashboard for monitoring the hybrid overnight run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _load_status(status_path: Path) -> dict:
    if not status_path.exists():
        raise FileNotFoundError(f"Status file not found: {status_path}")
    return json.loads(status_path.read_text(encoding="utf-8"))


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_duration(started_at: datetime | None, ended_at: datetime | None = None) -> str:
    if started_at is None:
        return "-"
    end_value = ended_at or datetime.now()
    delta = end_value - started_at
    total_seconds = max(int(delta.total_seconds()), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _dataset_stats(dataset_path: Path) -> dict:
    if not dataset_path.exists():
        return {
            "exists": False,
            "rows": 0,
            "successful_rows": 0,
            "failed_rows": 0,
            "last_sample_id": None,
        }

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return {
            "exists": True,
            "rows": 0,
            "successful_rows": 0,
            "failed_rows": 0,
            "last_sample_id": None,
            "error": str(e),
        }

    success_count = int((df["success"] == True).sum()) if "success" in df.columns else 0
    last_sample_id = None
    if "sample_id" in df.columns and len(df) > 0:
        last_sample_id = int(pd.to_numeric(df["sample_id"], errors="coerce").fillna(0).max())

    return {
        "exists": True,
        "rows": int(len(df)),
        "successful_rows": success_count,
        "failed_rows": int(len(df) - success_count),
        "last_sample_id": last_sample_id,
    }


def _print_header(title: str) -> None:
    print("=" * 90)
    print(title)
    print("=" * 90)


def _render(status: dict, dataset_stats: dict, status_path: Path) -> None:
    _clear_screen()

    started_at = _parse_time(status.get("started_at"))
    deadline = _parse_time(status.get("deadline"))
    finished_at = _parse_time(status.get("finished_at"))
    current_stage_started_at = _parse_time(status.get("current_stage_started_at"))

    _print_header("HYBRID RUN DASHBOARD")
    print(f"Status file:     {status_path}")
    print(f"Started:         {status.get('started_at', '-')}")
    print(f"Deadline:        {status.get('deadline', '-')}")
    print(f"Elapsed:         {_format_duration(started_at, finished_at)}")

    if deadline and not finished_at:
        remaining_seconds = int((deadline - datetime.now()).total_seconds())
        if remaining_seconds < 0:
            remaining_seconds = 0
        hours, rem = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Time remaining:  {hours:02d}:{minutes:02d}:{seconds:02d}")
    else:
        print("Time remaining:  -")

    print(f"Cycles complete: {status.get('cycles_completed', 0)}")
    print(f"Current stage:   {status.get('current_stage') or '-'}")
    print(f"Stage elapsed:   {_format_duration(current_stage_started_at) if status.get('current_stage') else '-'}")
    print("")

    _print_header("DATASET")
    print(f"Dataset path:    {status.get('dataset_path', '-')}")
    print(f"Rows:            {dataset_stats.get('rows', 0)}")
    print(f"Successful:      {dataset_stats.get('successful_rows', 0)}")
    print(f"Failed:          {dataset_stats.get('failed_rows', 0)}")
    print(f"Last sample id:  {dataset_stats.get('last_sample_id', '-')}")
    if dataset_stats.get("error"):
        print(f"Read error:      {dataset_stats['error']}")
    print("")

    _print_header("MODEL OUTPUT")
    print(f"Model dir:       {status.get('trained_models_dir', '-')}")
    print(f"Model root:      {status.get('model_root_name', '-')}")
    print("")

    _print_header("CURRENT COMMAND")
    current_command = status.get("current_command") or []
    if current_command:
        for part in current_command:
            print(part)
    else:
        print("-")
    print("")

    _print_header("RECENT STAGES")
    stages = status.get("stages", [])
    if not stages:
        print("-")
    else:
        for stage in stages[-8:]:
            dataset_snapshot = stage.get("dataset_stats", {})
            print(
                f"{stage.get('finished_at', '-')} | "
                f"{stage.get('stage', '-')} | "
                f"rc={stage.get('return_code', '-')} | "
                f"rows={dataset_snapshot.get('rows', 0)} | "
                f"success={dataset_snapshot.get('successful_rows', 0)}"
            )
    print("")

    if finished_at:
        _print_header("RUN COMPLETE")
        print(f"Finished at:     {status.get('finished_at')}")
    else:
        _print_header("LIVE")
        print("Press Ctrl+C to stop watching. The hybrid run will continue.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch the hybrid overnight run from the terminal")
    parser.add_argument("--status-path", required=True, help="Path to hybrid_run_status.json")
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    args = parser.parse_args()

    status_path = Path(args.status_path).resolve()

    try:
        while True:
            status = _load_status(status_path)
            dataset_path = Path(status["dataset_path"])
            dataset_stats = _dataset_stats(dataset_path)
            _render(status, dataset_stats, status_path)

            if status.get("finished_at"):
                return 0

            time.sleep(max(args.refresh_seconds, 1.0))
    except KeyboardInterrupt:
        print("\nStopped watching.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
