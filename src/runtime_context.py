"""
Runtime helpers for per-model config and model artifact resolution.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from artifact_paths import build_ensemble_filename, infer_model_root_name, resolve_model_output_dir, slugify


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_config_path(config_path: str | None = None) -> Path | None:
    candidate = config_path or os.environ.get("MLPIEZOV2_CONFIG_PATH")
    if not candidate:
        return None
    path = Path(candidate).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def load_runtime_module(config_path: str | None = None):
    resolved = resolve_config_path(config_path)
    if resolved is None:
        return None

    spec = importlib.util.spec_from_file_location("mlpiezov2_runtime_config", resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {resolved}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_runtime_config(config_path: str | None = None) -> tuple[dict, dict, dict, Path | None]:
    module = load_runtime_module(config_path)
    if module is None:
        from config import CONTROL_POINTS, DATASET, GEOSTUDIO  # type: ignore
        return CONTROL_POINTS, DATASET, GEOSTUDIO, None

    return module.CONTROL_POINTS, module.DATASET, module.GEOSTUDIO, Path(module.__file__).resolve()


def resolve_model_dir(model_dir: str | None = None) -> Path:
    candidate = model_dir or os.environ.get("MLPIEZOV2_MODEL_DIR") or "models"
    return Path(candidate).resolve()


def resolve_model_name(model_name: str | None = None, model_dir: str | Path | None = None) -> str:
    explicit = model_name or os.environ.get("MLPIEZOV2_MODEL_NAME")
    if explicit:
        return slugify(explicit)
    return infer_model_root_name(model_dir or "model")


def resolve_dataset_dir(dataset_dir: str | None = None, dataset_config: dict | None = None) -> Path:
    if dataset_dir:
        return Path(dataset_dir).resolve()
    env_candidate = os.environ.get("MLPIEZOV2_DATASET_DIR")
    if env_candidate:
        return Path(env_candidate).resolve()
    if dataset_config and dataset_config.get("output_dir"):
        return Path(dataset_config["output_dir"]).resolve()
    return (project_root() / "data" / "datasets" / "default").resolve()


def resolve_ensemble_path(
    mode: str,
    model_dir: str | None = None,
    model_name: str | None = None,
) -> Path:
    resolved_dir = resolve_model_dir(model_dir)
    resolved_name = resolve_model_name(model_name, resolved_dir)
    output_dir = resolve_model_output_dir(resolved_dir, resolved_name)
    return output_dir / build_ensemble_filename(resolved_name, mode)
