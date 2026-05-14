#!/usr/bin/env python3
"""
Prepare GeoStudio analysis working copies from an edited phreatic input workbook,
and optionally execute them with GeoCMD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).parent
PROJECT_ROOT = PIPELINE_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model_workflow import prepare_models_from_workbook, run_prepared_models, write_run_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and optionally run GeoStudio analyses from an edited workbook")
    parser.add_argument("--catalog", required=True, help="Path to model_catalog.json")
    parser.add_argument("--inputs", required=True, help="Path to phreatic_surface_inputs.xlsx")
    parser.add_argument(
        "--working-root",
        default=str(PROJECT_ROOT / "output" / "prepared_model_runs"),
        help="Directory for prepared working copies",
    )
    parser.add_argument("--run", action="store_true", help="Run GeoCMD after preparing model copies")
    parser.add_argument("--geocmd", help="Path to geocmd.exe when using --run")
    parser.add_argument("--timeout", type=int, default=None, help="Optional GeoCMD timeout in seconds")
    parser.add_argument("--parallel", type=int, default=1, help="Number of analyses to run concurrently")
    args = parser.parse_args()

    print("=" * 70)
    print("WORKBOOK-DRIVEN MODEL PREPARATION")
    print("=" * 70)
    print(f"Catalog:      {args.catalog}")
    print(f"Inputs:       {args.inputs}")
    print(f"Working root: {args.working_root}")
    print(f"Run models:   {args.run}")
    print(f"Parallel:     {args.parallel}")
    print("=" * 70)

    prepared = prepare_models_from_workbook(args.catalog, args.inputs, args.working_root)
    print("\n" + "=" * 70)
    print(f"Prepared {len(prepared)} models")
    print("=" * 70)

    for item in prepared[:10]:
        print(f"  {item.section_name} | {item.model_name} | analyses={len(item.analyses)}")
    if len(prepared) > 10:
        print(f"  ... and {len(prepared) - 10} more")

    if not args.run:
        print("\nPreparation complete. No GeoCMD execution requested.")
        return 0

    if not args.geocmd:
        print("\n--geocmd is required when using --run")
        return 1

    print("\n" + "=" * 70)
    print("STARTING GEOSTUDIO EXECUTION")
    print("=" * 70)

    results = run_prepared_models(
        prepared,
        args.geocmd,
        timeout_seconds=args.timeout,
        max_parallel=args.parallel,
    )
    summary_path = write_run_summary(results, Path(args.working_root) / "run_summary.csv")

    success_count = sum(1 for result in results if result.success)
    failed_count = len(results) - success_count
    print("\nExecution complete")
    print(f"  Successful models: {success_count}/{len(results)}")
    print(f"  Failed models:     {failed_count}/{len(results)}")
    print(f"  Summary: {summary_path}")

    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
