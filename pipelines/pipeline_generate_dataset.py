#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Generation Pipeline - WITH PHYSICS VALIDATION
Checks upstream > downstream BEFORE running GeoStudio
Resamples if violated - no wasted runs!
"""

import os
import sys
import time
import signal
import importlib.util
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional
import numpy as np

# Setup paths
PIPELINE_DIR = Path(__file__).parent
PROJECT_ROOT = PIPELINE_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CONFIG_DIR))

print(f"✓ Project root: {PROJECT_ROOT}")
print(f"✓ Using src: {SRC_DIR}")
print(f"✓ Config dir: {CONFIG_DIR}")

from phreatic_generator import PhreaticSurfaceGenerator, PhreaticSurfaceSampler
from geostudio_interface import GeoStudioInterface, PhreaticSurfaceData
from dataset_manager import DatasetManager, ProgressTracker
from config_utils import (
    get_control_point_info_for_scripts,
    apply_synchronization,
    print_control_point_summary,
    validate_control_point_config
)

try:
    from config import CONFIG, CONTROL_POINTS, GEOSTUDIO, DATASET
    from config_utils import (
        get_control_point_info_for_scripts,
        apply_synchronization,
        print_control_point_summary,
        validate_control_point_config
    )
    print("✓ Loaded config")
except ImportError as e:
    print(f"❌ Cannot load config: {e}")
    sys.exit(1)


def load_runtime_config(config_path: str | None) -> None:
    """Load config either from the default module or an explicit file path."""
    global CONFIG, CONTROL_POINTS, GEOSTUDIO, DATASET

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

    CONFIG = module.CONFIG
    CONTROL_POINTS = module.CONTROL_POINTS
    GEOSTUDIO = module.GEOSTUDIO
    DATASET = module.DATASET
    print(f"Loaded config: {config_file}")


# ========================= PHYSICS VALIDATION =========================

def validate_flow_direction(control_y: np.ndarray, strict: bool = True) -> tuple[bool, str]:
    """
    Validate that phreatic surface follows flow direction.
    
    Args:
        control_y: Control point Y values
        strict: If True, enforce monotonic decrease
        
    Returns:
        (is_valid, reason)
    """
    if len(control_y) < 2:
        return True, "OK"
    
    if strict:
        # Check monotonic non-increasing
        for i in range(1, len(control_y)):
            if control_y[i] > control_y[i-1] + 0.01:  # Small tolerance
                return False, f"CP{i+1} ({control_y[i]:.1f}) > CP{i} ({control_y[i-1]:.1f})"
        return True, "OK"
    else:
        # Just check upstream > downstream
        if control_y[-1] > control_y[0]:
            return False, f"Downstream ({control_y[-1]:.1f}) > Upstream ({control_y[0]:.1f})"
        return True, "OK"


def check_hydraulic_gradient(control_x: np.ndarray, control_y: np.ndarray, 
                             max_gradient: float = 2.0) -> tuple[bool, str]:
    """
    Check hydraulic gradients are reasonable.
    
    Args:
        control_x: X coordinates
        control_y: Y coordinates  
        max_gradient: Maximum allowed gradient
        
    Returns:
        (is_valid, reason)
    """
    for i in range(len(control_x) - 1):
        dx = control_x[i+1] - control_x[i]
        dy = abs(control_y[i+1] - control_y[i])
        
        if dx > 0:
            gradient = dy / dx
            if gradient > max_gradient:
                return False, f"Gradient {gradient:.2f} > {max_gradient} between CP{i+1} and CP{i+2}"
    
    return True, "OK"


def validate_sample_physics(control_y: np.ndarray, control_x: np.ndarray, 
                           max_gradient: float = 2.0) -> tuple[bool, str]:
    """
    Complete physics validation before running GeoStudio.
    
    Returns:
        (is_valid, reason_if_invalid)
    """
    # Check 1: Flow direction
    valid, reason = validate_flow_direction(control_y, strict=True)
    if not valid:
        return False, f"Flow direction: {reason}"
    
    # Check 2: Hydraulic gradient
    valid, reason = check_hydraulic_gradient(control_x, control_y, max_gradient)
    if not valid:
        return False, f"Gradient: {reason}"
    
    return True, "OK"


def generate_valid_sample(sampler, control_x: np.ndarray, max_gradient: float,
                         sync_groups: list[list[int]] | None = None,
                         max_attempts: int = 100) -> tuple[Optional[np.ndarray], int]:
    """
    Generate a single valid sample (with resampling if needed).
    
    Args:
        sampler: PhreaticSurfaceSampler instance
        control_x: X coordinates
        max_gradient: Maximum gradient
        max_attempts: Max resampling attempts
        
    Returns:
        (valid_sample, n_attempts) or (None, n_attempts) if failed
    """
    for attempt in range(1, max_attempts + 1):
        # Generate candidate
        candidate = sampler.sample_random(1, seed=None)[0]  # Random for resampling
        candidate = apply_synchronization(candidate.reshape(1, -1), sync_groups or [])[0]

        is_valid, reason = validate_sample_physics(candidate, control_x, max_gradient)
        
        if is_valid:
            return candidate, attempt
    
    return None, max_attempts


# ========================= LOGGING =========================

def setup_logging(log_dir: str, mode: str) -> logging.Logger:
    """Setup logging."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"dataset_{mode}_{timestamp}.log"
    
    logger = logging.getLogger("DatasetGeneration")
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(ch)
    
    return logger


# ========================= GRACEFUL SHUTDOWN =========================

class GracefulShutdown:
    def __init__(self):
        self.shutdown_requested = False
    
    def __enter__(self):
        signal.signal(signal.SIGINT, self._handler)
        return self
    
    def __exit__(self, *args):
        pass
    
    def _handler(self, signum, frame):
        if not self.shutdown_requested:
            print("\n" + "="*70)
            print("SHUTDOWN REQUESTED - Finishing current sample...")
            print("="*70)
            self.shutdown_requested = True
        else:
            sys.exit(1)
    
    def should_stop(self):
        return self.shutdown_requested


# ========================= MAIN PIPELINE =========================

def generate_dataset(mode: str, n_samples: int, logger: logging.Logger) -> None:
    """Generate dataset with physics validation."""
    
    print("\n" + "="*70)
    print(f"DATASET GENERATION - {mode.upper()} MODE")
    print("="*70)
    print(f"Samples: {n_samples}")
    print("="*70)
    
    # Validate config
    valid, msg = validate_control_point_config(CONTROL_POINTS)
    if not valid:
        print(f"\n❌ Invalid config: {msg}")
        sys.exit(1)
    
    # Get control point info
    print(f"\n[SETUP] Control Points:")
    print_control_point_summary(CONTROL_POINTS)
    
    cp_info = get_control_point_info_for_scripts(CONTROL_POINTS)
    
    # Initialize
    print(f"\n[SETUP] Components...")
    
    phreatic_gen = PhreaticSurfaceGenerator(
        control_point_x=cp_info['x_coords'].tolist(),
        n_total_points=cp_info['interpolation']['n_total_points'],
        interpolation_method=cp_info['interpolation']['method'],
        enforce_flow_direction=CONTROL_POINTS['constraints']['enforce_flow_direction']
    )
    print(f"  ✓ Generator: {cp_info['count']} CPs → {cp_info['interpolation']['n_total_points']} points")
    
    sampler = PhreaticSurfaceSampler(
        y_mins=cp_info['y_mins'].tolist(),
        y_maxs=cp_info['y_maxs'].tolist(),
        y_baselines=cp_info['y_baselines'].tolist()
    )
    print(f"  ✓ Sampler")
    
    template_path = GEOSTUDIO['templates'][mode]
    geocmd_exe = Path(GEOSTUDIO['geocmd_location']) / 'geocmd.exe'
    
    geostudio = GeoStudioInterface(template_path, str(geocmd_exe), GEOSTUDIO['fos_column_name'])
    print(f"  ✓ GeoStudio")
    
    dataset = DatasetManager(DATASET['output_dir'], mode, DATASET['version'])
    print(f"  ✓ Dataset manager")
    
    # Generate initial samples
    print(f"\n[STEP 1] Generating samples with physics validation...")
    
    method = DATASET['sampling']['method']
    seed = DATASET['sampling']['random_seed']
    
    if method == 'lhs':
        initial_samples = sampler.sample_lhs(n_samples, seed)
    else:
        initial_samples = sampler.sample_random(n_samples, seed)
    
    # Apply synchronization
    samples = apply_synchronization(initial_samples, cp_info['sync_groups'])
    
    # Validate and resample invalid ones
    validated_samples = []
    total_resamples = 0
    max_gradient = CONTROL_POINTS['constraints']['max_gradient']
    
    print(f"  Validating {len(samples)} samples...")
    
    for i, sample in enumerate(samples):
        is_valid, reason = validate_sample_physics(sample, cp_info['x_coords'], max_gradient)
        
        if is_valid:
            validated_samples.append(sample)
        else:
            # Resample until valid
            valid_sample, n_attempts = generate_valid_sample(
                sampler, cp_info['x_coords'], max_gradient, cp_info['sync_groups'], max_attempts=100
            )
            
            if valid_sample is not None:
                validated_samples.append(valid_sample)
                total_resamples += 1
            else:
                print(f"  ⚠️  Sample {i+1}: Could not generate valid sample after 50 attempts")
                # Use original anyway (will likely fail in GeoStudio)
                validated_samples.append(sample)
    
    samples = np.array(validated_samples)
    
    print(f"  ✓ Validation complete")
    print(f"    Valid initially: {len(samples) - total_resamples}")
    print(f"    Resampled: {total_resamples}")
    
    existing_total = dataset.n_total

    # Progress tracker
    progress = ProgressTracker(n_samples, DATASET['checkpointing']['frequency'])
    
    # Main loop
    print(f"\n[STEP 2] Running GeoStudio...")
    print("="*70)
    
    consecutive_failures = 0
    
    with GracefulShutdown() as shutdown:
        for i, control_y in enumerate(samples, start=1):
            sample_id = existing_total + i
            
            if shutdown.should_stop():
                break
            
            if consecutive_failures >= 20:
                print(f"\nSTOPPED: Too many failures")
                break
            
            print(f"\n{'─'*70}")
            print(f"Sample {i}/{n_samples} (global id {sample_id})")
            print(f"{'─'*70}")
            print(f"Control: {[f'{y:.1f}' for y in control_y]}")
            
            try:
                # Generate surface
                y_interp = phreatic_gen.generate(control_y)
                x_interp = phreatic_gen.get_x_coordinates()
                
                print(f"  ✓ Interpolated: Y=[{y_interp.min():.1f}, {y_interp.max():.1f}]")
                
                # Create data
                n_interp = len(y_interp)
                phreatic_data = PhreaticSurfaceData(
                    control_point_numbers=list(range(1, n_interp + 1)),
                    control_point_x=x_interp.tolist(),
                    control_point_y=control_y.tolist(),
                    interpolated_x=x_interp,
                    interpolated_y=y_interp
                )
                
                # Run
                result = geostudio.run_analysis(phreatic_data, sample_id)
                
                # Save
                dataset.add_sample(sample_id, control_y.tolist(), y_interp, result)
                
                if result.success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                
            except Exception as e:
                logger.error(f"Sample {i}: {e}")
                consecutive_failures += 1
            
            # Checkpoint
            if progress.should_checkpoint(i):
                dataset.save_checkpoint()
            
            # Progress
            if i % 5 == 0 or i == n_samples:
                progress.print_progress(i, dataset)
    
    # Save
    print("\n" + "="*70)
    print("SAVING...")
    print("="*70)
    
    dataset.save_final()
    
    stats = dataset.get_statistics()
    
    print(f"\n✅ COMPLETE - {mode.upper()}")
    print("="*70)
    print(f"Total: {stats['n_total']}")
    print(f"Success: {stats['n_success']} ({stats['success_rate']:.1f}%)")
    print(f"Resampled for physics: {total_resamples}")
    
    if 'fos' in stats:
        print(f"\nFOS: [{stats['fos']['min']:.3f}, {stats['fos']['max']:.3f}]")
    
    print("="*70)


def setup_logging(log_dir: str, mode: str) -> logging.Logger:
    """Setup logging."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"dataset_{mode}_{timestamp}.log"
    
    logger = logging.getLogger("DatasetGen")
    logger.setLevel(logging.INFO)
    
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(ch)
    
    return logger


def main():
    """Main."""
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['shallow', 'deep'])
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--config-path', type=str, default=None)
    
    args = parser.parse_args()
    load_runtime_config(args.config_path)
    
    logger = setup_logging('logs', args.mode)
    
    logger.info("="*60)
    logger.info(f"DATASET GENERATION - {args.mode.upper()}")
    logger.info("="*60)
    
    try:
        generate_dataset(args.mode, args.n_samples, logger)
        return 0
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        logger.error(f"Failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
