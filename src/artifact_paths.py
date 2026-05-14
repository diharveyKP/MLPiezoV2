"""
Helpers for consistent model-rooted artifact naming.
"""

from __future__ import annotations

import re
from pathlib import Path


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return cleaned.lower() or "model"


def infer_model_root_name(dataset_dir: str | Path, fallback: str = "model") -> str:
    dataset_path = Path(dataset_dir)
    return slugify(dataset_path.name or fallback)


def resolve_model_output_dir(base_output: str | Path, model_root_name: str) -> Path:
    return Path(base_output) / slugify(model_root_name)


def build_ensemble_filename(model_root_name: str, mode: str) -> str:
    return f"ensemble_{slugify(model_root_name)}_{mode}.pkl"
