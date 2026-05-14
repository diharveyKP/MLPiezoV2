"""
Configuration Utilities
Helper functions to extract control point info from config
Makes all scripts automatically adapt to config changes
"""

import numpy as np
from typing import List, Tuple, Dict, Any


def get_control_point_count(config: Dict) -> int:
    """Get number of control points."""
    return config['count']


def get_control_point_x(config: Dict) -> np.ndarray:
    """Get X coordinates as array."""
    return np.array(config['x_coordinates'])


def get_control_point_names(config: Dict) -> List[str]:
    """Get control point names."""
    return config['names']


def get_control_point_ranges(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get Y ranges for all control points.
    
    Returns:
        (y_mins, y_maxs, y_baselines) as numpy arrays
    """
    y_mins = np.array([r['min'] for r in config['y_ranges']])
    y_maxs = np.array([r['max'] for r in config['y_ranges']])
    y_baselines = np.array([r['baseline'] for r in config['y_ranges']])
    
    return y_mins, y_maxs, y_baselines


def get_sync_groups(config: Dict) -> List[List[int]]:
    """Get synchronization groups (indices to synchronize)."""
    return config.get('sync_groups', [])


def get_interpolation_config(config: Dict) -> Dict[str, Any]:
    """Get interpolation settings."""
    return config['interpolation']


def apply_synchronization(samples: np.ndarray, sync_groups: List[List[int]]) -> np.ndarray:
    """
    Apply synchronization rules to samples.
    
    Args:
        samples: Array of shape (n_samples, n_control_points)
        sync_groups: List of index groups to synchronize
        
    Returns:
        Synchronized samples
    """
    samples_sync = samples.copy()
    
    for group in sync_groups:
        if len(group) < 2:
            continue
        
        # Set all indices in group to first value
        first_idx = group[0]
        for idx in group[1:]:
            samples_sync[:, idx] = samples_sync[:, first_idx]
    
    return samples_sync


def get_feature_names_for_ml(config: Dict) -> List[str]:
    """
    Get feature names for ML models.
    
    Returns:
        List like ['control_y_1', 'control_y_2', ...]
    """
    n = config['count']
    return [f'control_y_{i+1}' for i in range(n)]


def get_bounds_for_optimization(config: Dict) -> List[Tuple[float, float]]:
    """
    Get bounds for optimization (inverse design, active learning).
    
    Returns:
        List of (min, max) tuples for each control point
    """
    return [(r['min'], r['max']) for r in config['y_ranges']]


def validate_control_point_config(config: Dict) -> Tuple[bool, str]:
    """
    Validate control point configuration.
    
    Returns:
        (is_valid, error_message)
    """
    required_keys = ['count', 'x_coordinates', 'names', 'y_ranges', 'interpolation']
    
    for key in required_keys:
        if key not in config:
            return False, f"Missing required key: {key}"
    
    n = config['count']
    
    if len(config['x_coordinates']) != n:
        return False, f"x_coordinates length ({len(config['x_coordinates'])}) != count ({n})"
    
    if len(config['names']) != n:
        return False, f"names length ({len(config['names'])}) != count ({n})"
    
    if len(config['y_ranges']) != n:
        return False, f"y_ranges length ({len(config['y_ranges'])}) != count ({n})"
    
    # Validate ranges
    for i, r in enumerate(config['y_ranges']):
        if r['min'] >= r['max']:
            return False, f"CP{i+1}: min >= max"
        if not (r['min'] <= r['baseline'] <= r['max']):
            return False, f"CP{i+1}: baseline outside range"
    
    # Validate sync groups
    if 'sync_groups' in config:
        for group in config['sync_groups']:
            for idx in group:
                if idx < 0 or idx >= n:
                    return False, f"Sync group index {idx} out of range [0, {n-1}]"
    
    return True, "OK"


def print_control_point_summary(config: Dict) -> None:
    """Print human-readable summary of control points."""
    
    print("\n" + "="*70)
    print("CONTROL POINT CONFIGURATION")
    print("="*70)
    print(f"Number of control points: {config['count']}")
    print(f"Interpolated points: {config['interpolation']['n_total_points']}")
    print(f"Model extent: X = [{config['interpolation']['x_min']:.1f}, {config['interpolation']['x_max']:.1f}] ft")
    
    print("\nControl Points:")
    for i in range(config['count']):
        x = config['x_coordinates'][i]
        name = config['names'][i]
        y_range = config['y_ranges'][i]
        print(f"  CP{i+1}: X={x:>10.2f} ft, Y=[{y_range['min']:.1f}, {y_range['max']:.1f}] ft - {name}")
    
    if 'sync_groups' in config and config['sync_groups']:
        print("\nSynchronization Groups:")
        for group in config['sync_groups']:
            cp_names = [f"CP{i+1}" for i in group]
            print(f"  {' = '.join(cp_names)}")
    
    print("="*70)


def get_control_point_info_for_scripts(control_points_config: Dict) -> Dict[str, Any]:
    """
    Get all control point info needed by scripts in one call.
    
    Returns dict with everything a script needs.
    """
    y_mins, y_maxs, y_baselines = get_control_point_ranges(control_points_config)
    
    return {
        'count': control_points_config['count'],
        'x_coords': get_control_point_x(control_points_config),
        'names': get_control_point_names(control_points_config),
        'y_mins': y_mins,
        'y_maxs': y_maxs,
        'y_baselines': y_baselines,
        'sync_groups': get_sync_groups(control_points_config),
        'bounds': get_bounds_for_optimization(control_points_config),
        'interpolation': get_interpolation_config(control_points_config),
        'feature_names': get_feature_names_for_ml(control_points_config)
    }

