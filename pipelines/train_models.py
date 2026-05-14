#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Training Pipeline
Train ensemble on actual Fort Knox dataset
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import pickle

# Add src to path
# Use relative path
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
PROJECT_ROOT = PIPELINE_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

print(f"✓ Using src: {SRC_DIR}")

from data_loader import FOSDataLoader, print_dataset_info
from ml_models import create_model, XGBOOST_AVAILABLE
from ensemble import FOSEnsemble, predict_with_confidence
from artifact_paths import build_ensemble_filename, infer_model_root_name, resolve_model_output_dir, slugify


# ========================= LOGGING =========================

def setup_logging() -> logging.Logger:
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    return logging.getLogger("ModelTraining")


# ========================= TRAINING PIPELINE =========================

def train_ensemble_pipeline(dataset_dir: str, mode: str = 'shallow', 
                           min_samples: int = 10) -> FOSEnsemble:
    """
    Complete training pipeline.
    
    Args:
        dataset_dir: Path to dataset directory
        mode: 'shallow' or 'deep'
        min_samples: Minimum samples required to train
        
    Returns:
        Trained FOSEnsemble
    """
    
    print("\n" + "="*70)
    print(f"🎓 TRAINING ENSEMBLE - {mode.upper()} MODE")
    print("="*70)
    
    # STEP 1: Load data
    print(f"\n[STEP 1] Loading dataset...")
    print(f"  Location: {dataset_dir}")
    
    loader = FOSDataLoader(dataset_dir)
    
    try:
        data_splits = loader.prepare_for_training(
            mode=mode,
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
    except FileNotFoundError as e:
        print(f"\n❌ Dataset not found!")
        print(f"   {e}")
        print(f"\nGenerate data first:")
        print(f"  python pipelines\\pipeline_generate_dataset.py --mode {mode} --n_samples 50")
        sys.exit(1)
    
    X_train, y_train = data_splits['train']
    X_val, y_val = data_splits['val']
    X_test, y_test = data_splits['test']
    
    # Check minimum samples
    if len(X_train) < min_samples:
        print(f"\n⚠️  WARNING: Only {len(X_train)} training samples!")
        print(f"   Recommended: At least {min_samples} samples for reliable training")
        response = input(f"\n   Continue anyway? (y/n): ").lower()
        if response != 'y':
            print("Cancelled. Generate more data first.")
            sys.exit(0)
    
    print(f"  ✓ Loaded {len(X_train) + len(X_val) + len(X_test)} samples")
    print(f"    Train: {len(X_train)}")
    print(f"    Val:   {len(X_val)}")
    print(f"    Test:  {len(X_test)}")
    print(f"    FOS range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # STEP 2: Create ensemble
    print(f"\n[STEP 2] Building ensemble...")
    
    ensemble = FOSEnsemble()
    
    # Add models
    print(f"  Adding models to ensemble:")
    
    ensemble.add_model(create_model('linear'))
    print(f"    ✓ Linear (Ridge regression)")
    
    ensemble.add_model(create_model('random_forest', n_estimators=100, max_depth=10))
    print(f"    ✓ Random Forest (100 trees)")
    
    ensemble.add_model(create_model('gradient_boost', n_estimators=100, max_depth=5))
    print(f"    ✓ Gradient Boosting (100 trees)")
    
    if XGBOOST_AVAILABLE:
        ensemble.add_model(create_model('xgboost', n_estimators=100, max_depth=6))
        print(f"    ✓ XGBoost (100 trees)")
    else:
        print(f"    ⚠️  XGBoost not available (install: pip install xgboost)")
    
    ensemble.add_model(create_model('neural_net', hidden_layers=(64, 32, 16)))
    print(f"    ✓ Neural Network (64-32-16)")
    
    print(f"  ✓ Ensemble ready with {len(ensemble.models)} models")
    
    # STEP 3: Train ensemble
    print(f"\n[STEP 3] Training ensemble...")
    
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    print(f"  ✓ All models trained")
    
    # STEP 4: Evaluate on test set
    print(f"\n[STEP 4] Evaluating on test set...")
    
    # Ensemble predictions
    y_pred_mean, y_pred_std, confidence = ensemble.predict_with_uncertainty(X_test)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    mae = mean_absolute_error(y_test, y_pred_mean)
    r2 = r2_score(y_test, y_pred_mean)
    
    print(f"\n  📊 ENSEMBLE PERFORMANCE:")
    print(f"     RMSE: {rmse:.4f}")
    print(f"     MAE:  {mae:.4f}")
    print(f"     R²:   {r2:.4f}")
    print(f"     Avg Uncertainty: {y_pred_std.mean():.4f}")
    print(f"     Avg Confidence:  {confidence.mean():.1%}")
    
    # Analyze predictions
    errors = np.abs(y_test - y_pred_mean)
    print(f"\n  📈 ERROR ANALYSIS:")
    print(f"     Max error: {errors.max():.4f}")
    print(f"     95th percentile error: {np.percentile(errors, 95):.4f}")
    print(f"     Samples with error > 0.05: {np.sum(errors > 0.05)}/{len(errors)}")
    
    # Critical region analysis (FOS < 1.2)
    critical_mask = y_test < 1.2
    if np.any(critical_mask):
        critical_rmse = np.sqrt(mean_squared_error(y_test[critical_mask], y_pred_mean[critical_mask]))
        print(f"\n  ⚠️  CRITICAL REGION (FOS < 1.2):")
        print(f"     Samples: {np.sum(critical_mask)}")
        print(f"     RMSE: {critical_rmse:.4f}")
    
    # STEP 5: Example predictions
    print(f"\n[STEP 5] Example predictions with uncertainty...")
    
    for i in range(min(3, len(X_test))):
        result = predict_with_confidence(ensemble, X_test[i])
        actual = y_test[i]
        error = abs(result.fos_mean - actual)
        
        print(f"\n  Sample {i+1}:")
        print(f"    Control points: {X_test[i]}")
        print(f"    {result}")
        print(f"    Actual FOS: {actual:.3f}")
        print(f"    Error: {error:.4f}")
        print(f"    Failure prob: {result.get_failure_probability(1.0):.1%}")
    
    return ensemble


def save_ensemble(ensemble: FOSEnsemble, output_path: str, mode: str, model_root_name: str) -> Path | None:
    """Save trained ensemble to disk."""
    output_dir = resolve_model_output_dir(output_path, model_root_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / build_ensemble_filename(model_root_name, mode)
    
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(ensemble, f)
        
        print(f"\n💾 Model saved: {output_file}")
        print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
        return output_file
        
    except Exception as e:
        print(f"\n❌ Failed to save model: {e}")


def load_ensemble(model_path: str, mode: str, model_root_name: str | None = None) -> FOSEnsemble:
    """Load trained ensemble from disk."""
    resolved_model_root = slugify(model_root_name) if model_root_name else infer_model_root_name(model_path)
    model_file = resolve_model_output_dir(model_path, resolved_model_root) / build_ensemble_filename(resolved_model_root, mode)
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    
    with open(model_file, 'rb') as f:
        ensemble = pickle.load(f)
    
    print(f"✓ Loaded ensemble: {model_file}")
    print(f"  Models: {len(ensemble.models)}")
    
    return ensemble

def analyze_confidence_intervals(ensemble, X_test, y_test, fos_ranges=None):
    """
    Analyze confidence interval calibration across FOS ranges.
    
    Args:
        ensemble: Trained ensemble
        X_test: Test features
        y_test: True FOS values
        fos_ranges: List of (min, max) tuples for FOS ranges to analyze
    """
    if fos_ranges is None:
        fos_ranges = [(0.0, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 2.0)]
    
    print("\n" + "="*70)
    print("🎯 CONFIDENCE INTERVAL CALIBRATION ANALYSIS")
    print("="*70)
    
    # Get predictions with uncertainty
    y_pred, y_std, confidence = ensemble.predict_with_uncertainty(X_test)
    
    # Get quantiles for different confidence levels
    quantiles_80 = ensemble.predict_quantiles(X_test, [0.10, 0.90])  # 80% CI
    quantiles_90 = ensemble.predict_quantiles(X_test, [0.05, 0.95])  # 90% CI
    quantiles_95 = ensemble.predict_quantiles(X_test, [0.025, 0.975])  # 95% CI
    
    # Overall calibration
    print("\n📊 OVERALL CALIBRATION (All FOS ranges):")
    print("─"*70)
    
    # Check coverage
    in_80 = ((y_test >= quantiles_80[0.10]) & (y_test <= quantiles_80[0.90])).sum()
    in_90 = ((y_test >= quantiles_90[0.05]) & (y_test <= quantiles_90[0.95])).sum()
    in_95 = ((y_test >= quantiles_95[0.025]) & (y_test <= quantiles_95[0.975])).sum()
    
    n_test = len(y_test)
    
    print(f"  80% CI: Covers {in_80}/{n_test} = {in_80/n_test*100:.1f}% (target: 80%)")
    print(f"  90% CI: Covers {in_90}/{n_test} = {in_90/n_test*100:.1f}% (target: 90%)")
    print(f"  95% CI: Covers {in_95}/{n_test} = {in_95/n_test*100:.1f}% (target: 95%)")
    
    # Calibration quality
    print(f"\n  Mean CI widths:")
    print(f"    80%: ±{(quantiles_80[0.90] - quantiles_80[0.10]).mean()/2:.4f}")
    print(f"    90%: ±{(quantiles_90[0.95] - quantiles_90[0.05]).mean()/2:.4f}")
    print(f"    95%: ±{(quantiles_95[0.975] - quantiles_95[0.025]).mean()/2:.4f}")
    
    # Analyze by FOS range
    print("\n" + "="*70)
    print("📍 CALIBRATION BY FOS RANGE:")
    print("="*70)
    
    for fos_min, fos_max in fos_ranges:
        # Find samples in this FOS range
        mask = (y_test >= fos_min) & (y_test < fos_max)
        n_in_range = mask.sum()
        
        if n_in_range < 3:
            continue  # Skip if too few samples
        
        print(f"\n🎯 FOS Range: [{fos_min:.1f}, {fos_max:.1f})")
        print("─"*70)
        print(f"  Samples in range: {n_in_range}")
        
        # Coverage in this range
        in_80_range = ((y_test[mask] >= quantiles_80[0.10][mask]) & 
                       (y_test[mask] <= quantiles_80[0.90][mask])).sum()
        in_90_range = ((y_test[mask] >= quantiles_90[0.05][mask]) & 
                       (y_test[mask] <= quantiles_90[0.95][mask])).sum()
        in_95_range = ((y_test[mask] >= quantiles_95[0.025][mask]) & 
                       (y_test[mask] <= quantiles_95[0.975][mask])).sum()
        
        print(f"\n  Coverage:")
        cov_80 = in_80_range/n_in_range*100
        cov_90 = in_90_range/n_in_range*100
        cov_95 = in_95_range/n_in_range*100
        
        print(f"    80% CI: {cov_80:.1f}% {self._calibration_symbol(cov_80, 80)}")
        print(f"    90% CI: {cov_90:.1f}% {self._calibration_symbol(cov_90, 90)}")
        print(f"    95% CI: {cov_95:.1f}% {self._calibration_symbol(cov_95, 95)}")
        
        # Mean errors and uncertainties
        errors = np.abs(y_test[mask] - y_pred[mask])
        uncertainties = y_std[mask]
        
        print(f"\n  Uncertainty Statistics:")
        print(f"    Mean prediction error: ±{errors.mean():.4f}")
        print(f"    Mean uncertainty (std): ±{uncertainties.mean():.4f}")
        print(f"    Mean confidence: {confidence[mask].mean()*100:.1f}%")
        
        # CI widths in this range
        width_80 = (quantiles_80[0.90][mask] - quantiles_80[0.10][mask]).mean()
        width_90 = (quantiles_90[0.95][mask] - quantiles_90[0.05][mask]).mean()
        width_95 = (quantiles_95[0.975][mask] - quantiles_95[0.025][mask]).mean()
        
        print(f"\n  Typical CI widths:")
        print(f"    80% CI: ±{width_80/2:.4f} → FOS ± {width_80/2:.3f}")
        print(f"    90% CI: ±{width_90/2:.4f} → FOS ± {width_90/2:.3f}")
        print(f"    95% CI: ±{width_95/2:.4f} → FOS ± {width_95/2:.3f}")
    
    print("\n" + "="*70)

def _calibration_symbol(actual, target):
    """Return symbol for calibration quality."""
    diff = abs(actual - target)
    if diff < 5:
        return "✓ Excellent"
    elif diff < 10:
        return "✓ Good"
    elif diff < 15:
        return "⚠ Fair"
    else:
        return "❌ Poor"

# ========================= MAIN =========================

def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FOS ensemble model")
    parser.add_argument('--dataset', type=str,
                        default=str((PROJECT_ROOT / 'data' / 'datasets' / 'default').resolve()),
                        help='Dataset directory')
    parser.add_argument('--mode', type=str, default='shallow',
                       choices=['shallow', 'deep'],
                       help='Which mode to train')
    parser.add_argument('--output', type=str, default='models',
                       help='Where to save trained model')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Optional model root name for artifact naming')
    parser.add_argument('--min_samples', type=int, default=20,
                       help='Minimum samples required')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    model_root_name = slugify(args.model_name) if args.model_name else infer_model_root_name(args.dataset)
    
    print("="*70)
    print("🎓 FOS ENSEMBLE TRAINING PIPELINE")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    print(f"Model root: {model_root_name}")
    print("="*70)
    
    # Show dataset info
    print_dataset_info(args.dataset, args.mode)
    
    # Train ensemble
    try:
        ensemble = train_ensemble_pipeline(
            dataset_dir=args.dataset,
            mode=args.mode,
            min_samples=args.min_samples
        )
        
        # Save model
        saved_model_path = save_ensemble(ensemble, args.output, args.mode, model_root_name)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\nTrained ensemble saved to: {saved_model_path}")
        print(f"\nTo use the model:")
        print(f"  from ensemble import load_ensemble")
        print(f"  ensemble = load_ensemble('{args.output}', '{args.mode}', '{model_root_name}')")
        print(f"  prediction = ensemble.predict([[1565, 1550, 1485, 1480]])")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
