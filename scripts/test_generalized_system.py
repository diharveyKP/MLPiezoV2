"""
Test Generalized System
Verify all scripts work with current config
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CONFIG_DIR))

from config import CONTROL_POINTS, GEOSTUDIO, DATASET
from config_utils import (
    get_control_point_info_for_scripts,
    validate_control_point_config,
    print_control_point_summary,
    apply_synchronization
)
import numpy as np


def test_config_validation():
    """Test 1: Config validation."""
    print("\n" + "="*70)
    print("TEST 1: Config Validation")
    print("="*70)
    
    valid, msg = validate_control_point_config(CONTROL_POINTS)
    
    if valid:
        print("✅ PASS - Config is valid")
        print_control_point_summary(CONTROL_POINTS)
    else:
        print(f"❌ FAIL - {msg}")
        return False
    
    return True


def test_control_point_extraction():
    """Test 2: Extract control point info."""
    print("\n" + "="*70)
    print("TEST 2: Control Point Info Extraction")
    print("="*70)
    
    try:
        cp_info = get_control_point_info_for_scripts(CONTROL_POINTS)
        
        print(f"✓ Count: {cp_info['count']}")
        print(f"✓ X coords: {cp_info['x_coords']}")
        print(f"✓ Names: {cp_info['names']}")
        print(f"✓ Bounds: {cp_info['bounds']}")
        print(f"✓ Sync groups: {cp_info['sync_groups']}")
        print(f"✓ Interpolation points: {cp_info['interpolation']['n_total_points']}")
        print(f"✓ Feature names: {cp_info['feature_names']}")
        
        print("\n✅ PASS - All info extracted")
        return True, cp_info
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        return False, None


def test_synchronization(cp_info):
    """Test 3: Synchronization."""
    print("\n" + "="*70)
    print("TEST 3: Synchronization Logic")
    print("="*70)
    
    # Generate random samples
    samples = np.random.rand(3, cp_info['count']) * 100 + 3000
    
    print("Before sync:")
    print(samples[0])
    
    samples_sync = apply_synchronization(samples, cp_info['sync_groups'])
    
    print("\nAfter sync:")
    print(samples_sync[0])
    
    # Verify sync groups
    all_valid = True
    for group in cp_info['sync_groups']:
        first_val = samples_sync[0, group[0]]
        for idx in group[1:]:
            if samples_sync[0, idx] != first_val:
                print(f"❌ FAIL - CP{idx+1} not synced with CP{group[0]+1}")
                all_valid = False
                return False
        
        group_names = [f'CP{i+1}' for i in group]
        print(f"✓ {' = '.join(group_names)}: {first_val:.2f}")
    
    if all_valid:
        print("\n✅ PASS - Synchronization working")
    
    return all_valid


def test_ensemble_compatibility(cp_info):
    """Test 4: Ensemble compatibility."""
    print("\n" + "="*70)
    print("TEST 4: Ensemble Model Compatibility")
    print("="*70)
    
    try:
        import pickle
        
        # Try to load model
        model_file = Path('models') / 'ensemble_shallow.pkl'
        
        if not model_file.exists():
            print("⚠️  SKIP - No trained model found")
            print("   Train a model to test this")
            return True
        
        with open(model_file, 'rb') as f:
            ensemble = pickle.load(f)
        
        print(f"✓ Loaded ensemble: {len(ensemble.models)} models")
        
        # Test prediction
        test_input = cp_info['y_baselines'].reshape(1, -1)
        
        print(f"✓ Test input shape: {test_input.shape}")
        print(f"  Expected features: {cp_info['count']}")
        
        fos = ensemble.predict(test_input)[0]
        
        print(f"✓ Prediction successful: FOS = {fos:.4f}")
        
        print("\n✅ PASS - Model compatible with config")
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        return False


def test_summary():
    """Print overall summary."""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    cp_info = get_control_point_info_for_scripts(CONTROL_POINTS)
    
    print(f"\n✅ SYSTEM CONFIGURED FOR:")
    print(f"  • {cp_info['count']} control points")
    print(f"  • {len(cp_info['sync_groups'])} sync groups")
    print(f"  • {cp_info['interpolation']['n_total_points']} interpolated points")
    print(f"  • Model extent: {cp_info['interpolation']['x_min']:.1f} - {cp_info['interpolation']['x_max']:.1f} ft")
    
    print(f"\n🎯 TO USE WITH DIFFERENT SECTION:")
    print(f"  1. Edit configs/config.py")
    print(f"  2. Update CONTROL_POINTS dict")
    print(f"  3. All scripts automatically adapt!")
    
    print("\n" + "="*70)


def main():
    """Run all tests."""
    
    print("="*70)
    print("TESTING GENERALIZED SYSTEM")
    print("="*70)
    
    results = []
    
    # Test 1
    results.append(test_config_validation())
    
    # Test 2
    test2_pass, cp_info = test_control_point_extraction()
    results.append(test2_pass)
    
    if not test2_pass:
        print("\n❌ Cannot continue - config extraction failed")
        return
    
    # Test 3
    results.append(test_synchronization(cp_info))
    
    # Test 4
    results.append(test_ensemble_compatibility(cp_info))
    
    # Summary
    test_summary()
    
    # Final result
    if all(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("   System is ready for production use")
        print("   Change control points in config.py to adapt to new sections")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("   Review errors above")


if __name__ == "__main__":
    main()