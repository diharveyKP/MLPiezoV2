#!/usr/bin/env python3
"""
Build a Section -> Model -> Analysis catalog from GeoStudio XML or GSZ files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).parent
PROJECT_ROOT = PIPELINE_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model_catalog import discover_models, export_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover GeoStudio model structure and export configs/catalogs")
    parser.add_argument("--root", required=True, help="Root folder containing extracted XML models and/or GSZ files")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "output" / "model_catalog"),
        help="Output folder for generated catalog files",
    )
    args = parser.parse_args()

    root_path = Path(args.root)
    output_path = Path(args.output)

    print("=" * 70)
    print("GEOSTUDIO MODEL CATALOG DISCOVERY")
    print("=" * 70)
    print(f"Root:   {root_path}")
    print(f"Output: {output_path}")
    print("=" * 70)

    models = discover_models(root_path)

    print(f"\nDiscovered {len(models)} models")
    print(f"Total analyses: {sum(len(model.analyses) for model in models)}")

    outputs = export_catalog(models, output_path)

    print("\nGenerated outputs:")
    print(f"  Catalog JSON: {outputs['catalog_json']}")
    print(f"  Excel file:   {outputs['workbook']}")
    print(f"  Inputs XLSX:  {outputs['inputs_workbook']}")
    print(f"  Config dir:   {outputs['configs_dir']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
