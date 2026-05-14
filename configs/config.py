r"""
Default repository-local configuration for MLPiezoV2.

This file is intentionally generic. For real model work, prefer generating
and using a per-model config via:
    python pipelines\bootstrap_single_model.py --model-path <model.gsz>
Then pass the resulting config with:
    --config-path <workspace>\model_config.py
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


MODEL_INFO = {
    "project": "MLPiezoV2",
    "section": "default",
    "description": "Default repository-local configuration",
    "units": "Imperial (feet)",
    "version": "v2",
}


CONTROL_POINTS = {
    "count": 5,
    "x_coordinates": [0.0, 450.0, 900.0, 1350.0, 1800.0],
    "names": ["CP1", "CP2", "CP3", "CP4", "CP5"],
    "y_ranges": [
        {"min": 3000.0, "max": 3090.0, "baseline": 3090.0},
        {"min": 2980.0, "max": 3075.0, "baseline": 3075.0},
        {"min": 2950.0, "max": 3050.0, "baseline": 3050.0},
        {"min": 2900.0, "max": 3000.0, "baseline": 3000.0},
        {"min": 2850.0, "max": 2950.0, "baseline": 2950.0},
    ],
    "sync_groups": [],
    "interpolation": {
        "n_total_points": 100,
        "method": "pchip",
        "x_min": 0.0,
        "x_max": 1800.0,
    },
    "constraints": {
        "enforce_flow_direction": True,
        "max_gradient": 2.0,
        "enforce_below_ground": True,
        "ground_clearance": 2.0,
    },
}


GEOSTUDIO = {
    "geocmd_location": "C:/Program Files/Seequent/GeoStudio 2024.1/Bin",
    "templates": {
        "shallow": str(PROJECT_ROOT / "models" / "templates" / "shallow.xml"),
        "deep": str(PROJECT_ROOT / "models" / "templates" / "deep.xml"),
    },
    "fos_column_name": "SlipFOS",
    "solver": {
        "retries": 3,
        "timeout_seconds": 300,
    },
}


DATASET = {
    "output_dir": str(PROJECT_ROOT / "data" / "datasets" / "default"),
    "version": "v2",
    "description": "Default dataset output",
    "sampling": {
        "method": "lhs",
        "random_seed": 42,
    },
    "checkpointing": {
        "frequency": 10,
        "enabled": True,
    },
}


ML_CONFIG = {
    "train_test_split": {
        "test_size": 0.2,
        "val_size": 0.1,
        "random_state": 42,
    },
    "models": {
        "linear": {"alpha": 10.0},
        "random_forest": {"n_estimators": 300, "max_depth": 12},
        "gradient_boost": {"n_estimators": 300, "max_depth": 3},
        "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
        "neural_net": {"hidden_layers": (128, 64, 32)},
    },
    "ensemble": {"use_weighted": True},
}


CONFIG = {
    "geostudio": {
        "geocmd_location": GEOSTUDIO["geocmd_location"],
        "template_shallow": GEOSTUDIO["templates"]["shallow"],
        "template_deep": GEOSTUDIO["templates"]["deep"],
        "fos_column_name": GEOSTUDIO["fos_column_name"],
        "retries": GEOSTUDIO["solver"]["retries"],
        "timeout_seconds": GEOSTUDIO["solver"]["timeout_seconds"],
    },
    "phreatic": {
        "control_points": {
            "points": [
                {
                    "number": i + 1,
                    "x": CONTROL_POINTS["x_coordinates"][i],
                    "y_min": CONTROL_POINTS["y_ranges"][i]["min"],
                    "y_max": CONTROL_POINTS["y_ranges"][i]["max"],
                    "y_baseline": CONTROL_POINTS["y_ranges"][i]["baseline"],
                    "name": CONTROL_POINTS["names"][i],
                }
                for i in range(CONTROL_POINTS["count"])
            ]
        },
        "interpolation": CONTROL_POINTS["interpolation"],
        "constraints": CONTROL_POINTS["constraints"],
    },
    "dataset": {
        "n_samples": 100,
        "sampling_method": DATASET["sampling"]["method"],
        "output_dir": DATASET["output_dir"],
        "version": DATASET["version"],
        "description": DATASET["description"],
        "checkpoint_frequency": DATASET["checkpointing"]["frequency"],
        "random_seed": DATASET["sampling"]["random_seed"],
    },
    "error_handling": {"max_consecutive_failures": 20},
    "logging": {"log_dir": str(PROJECT_ROOT / "logs"), "level": "INFO"},
    "system": {"precision": 0.01, "rounding_decimals": 2},
}
