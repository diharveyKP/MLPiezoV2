"""
Ensemble Model with Uncertainty Quantification
Combines multiple models for robust predictions with confidence estimates
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FOSEnsemble:
    """
    Multi-model ensemble for FOS prediction with uncertainty.
    
    Features:
    - Combines predictions from multiple models
    - Estimates uncertainty from model disagreement
    - Provides confidence scores
    - Identifies high-risk predictions
    """
    
    def __init__(self, models: List = None):
        """
        Args:
            models: List of FOSModelBase instances
        """
        self.models = models or []
        self.is_fitted = False
        self.model_weights = None
        
        logger.info(f"Ensemble initialized with {len(self.models)} models")
    
    def add_model(self, model) -> None:
        """Add a model to the ensemble."""
        self.models.append(model)
        logger.info(f"Added model: {model.name}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> 'FOSEnsemble':
        """
        Train all models in ensemble.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (for weighted ensemble)
            y_val: Validation targets (for weighted ensemble)
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble! Add models first.")
        
        logger.info(f"Training ensemble of {len(self.models)} models...")
        
        # Train each model
        for i, model in enumerate(self.models, 1):
            print(f"  [{i}/{len(self.models)}] Training {model.name}...")
            try:
                model.fit(X, y)
                print(f"      ✓ Complete")
            except Exception as e:
                logger.error(f"Failed to train {model.name}: {e}")
                print(f"      ❌ Failed: {e}")
        
        # Calculate weights based on validation performance (if provided)
        if X_val is not None and y_val is not None:
            self._calculate_weights(X_val, y_val)
        else:
            # Equal weights
            self.model_weights = np.ones(len(self.models)) / len(self.models)
        
        self.is_fitted = True
        logger.info("Ensemble training complete")
        
        return self
    
    def _calculate_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Calculate model weights based on validation performance."""
        from sklearn.metrics import mean_squared_error
        
        errors = []
        for model in self.models:
            try:
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                errors.append(mse)
            except:
                errors.append(1e10)  # Large error if prediction fails
        
        # Inverse MSE weighting (lower error = higher weight)
        errors = np.array(errors)
        inv_errors = 1.0 / (errors + 1e-10)
        self.model_weights = inv_errors / inv_errors.sum()
        
        logger.info("Model weights calculated:")
        for model, weight in zip(self.models, self.model_weights):
            logger.info(f"  {model.name}: {weight:.3f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Simple prediction (mean of ensemble)."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet!")
        
        predictions = self._get_all_predictions(X)
        
        # Weighted average
        if self.model_weights is not None:
            return np.average(predictions, axis=0, weights=self.model_weights)
        else:
            return np.mean(predictions, axis=0)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates.
        
        Returns:
            (mean, std, confidence) where:
                mean: Ensemble mean prediction [n_samples]
                std: Standard deviation across models [n_samples] (UNCERTAINTY!)
                confidence: Confidence score 0-1 [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet!")
        
        # Get predictions from all models
        predictions = self._get_all_predictions(X)  # [n_models, n_samples]
        
        # Ensemble statistics
        if self.model_weights is not None:
            mean_pred = np.average(predictions, axis=0, weights=self.model_weights)
            variance = np.average((predictions - mean_pred) ** 2, axis=0, weights=self.model_weights)
            std_pred = np.sqrt(variance)
        else:
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
        
        # Confidence score (inverse of normalized std)
        # Low std = high confidence, high std = low confidence
        max_std = 0.3  # Assume std > 0.3 is very uncertain
        confidence = np.clip(1.0 - (std_pred / max_std), 0.0, 1.0)
        
        return mean_pred, std_pred, confidence
    
    def predict_quantiles(self, X: np.ndarray, quantiles: List[float] = [0.025, 0.975]) -> Dict[float, np.ndarray]:
        """
        Predict quantiles from ensemble distribution.
        
        Args:
            X: Input features
            quantiles: List of quantiles to compute (e.g., [0.025, 0.975] for 95% CI)
            
        Returns:
            Dict mapping quantile to predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet!")
        
        predictions = self._get_all_predictions(X)  # [n_models, n_samples]
        
        results = {}
        for q in quantiles:
            results[q] = np.quantile(predictions, q, axis=0)
        
        return results
    
    def _get_all_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all models."""
        predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Prediction failed for {model.name}: {e}")
                # Use mean prediction as fallback
                if predictions:
                    predictions.append(np.mean(predictions, axis=0))
        
        return np.array(predictions)
    
    def get_model_agreement(self, X: np.ndarray) -> np.ndarray:
        """
        Measure model agreement (inverse of uncertainty).
        
        Returns:
            Agreement score 0-1 for each sample (1 = all models agree perfectly)
        """
        _, std, _ = self.predict_with_uncertainty(X)
        
        # Convert std to agreement (0 std = 1.0 agreement)
        max_std = 0.3
        agreement = np.clip(1.0 - (std / max_std), 0.0, 1.0)
        
        return agreement
    
    def get_feature_importance_ensemble(self) -> Dict[str, np.ndarray]:
        """Get feature importance from all models that support it."""
        importances = {}
        
        for model in self.models:
            imp = model.get_feature_importance()
            if imp is not None:
                importances[model.name] = imp
        
        return importances


# ========================= PREDICTION RESULT CLASS =========================

class PredictionResult:
    """Container for prediction with uncertainty."""
    
    def __init__(self, fos_mean: float, fos_std: float, confidence: float,
                 ci_lower: float, ci_upper: float):
        self.fos_mean = fos_mean
        self.fos_std = fos_std
        self.confidence = confidence
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
    
    def __repr__(self):
        return (f"FOS = {self.fos_mean:.3f} ± {self.fos_std:.3f} "
                f"(95% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}], "
                f"Confidence: {self.confidence:.1%})")
    
    def is_failure_risk(self, threshold: float = 1.1) -> bool:
        """Check if lower confidence bound is below threshold."""
        return self.ci_lower < threshold
    
    def get_failure_probability(self, threshold: float = 1.0) -> float:
        """
        Estimate probability that true FOS < threshold.
        Assumes normal distribution.
        """
        from scipy.stats import norm
        z_score = (threshold - self.fos_mean) / (self.fos_std + 1e-10)
        return float(norm.cdf(z_score))


# ========================= CONVENIENCE FUNCTION =========================

def predict_with_confidence(ensemble: FOSEnsemble, 
                           control_points: np.ndarray) -> PredictionResult:
    """
    Make prediction with full confidence information.
    
    Args:
        ensemble: Fitted ensemble
        control_points: [4] or [n, 4] array of control point Y values
        
    Returns:
        PredictionResult with mean, std, confidence, intervals
    """
    # Ensure 2D
    if control_points.ndim == 1:
        control_points = control_points.reshape(1, -1)
    
    # Predict
    mean, std, conf = ensemble.predict_with_uncertainty(control_points)
    
    # 95% confidence interval (±2 std)
    ci_lower = mean - 2 * std
    ci_upper = mean + 2 * std
    
    # Return result for first sample
    return PredictionResult(
        fos_mean=float(mean[0]),
        fos_std=float(std[0]),
        confidence=float(conf[0]),
        ci_lower=float(ci_lower[0]),
        ci_upper=float(ci_upper[0])
    )


# ========================= TEST CODE =========================

if __name__ == "__main__":
    from ml_models import create_model
    
    print("="*70)
    print("🧪 ENSEMBLE TEST")
    print("="*70)
    
    # Create dummy data
    print("\n[1] Creating test data...")
    np.random.seed(42)
    X_train = np.random.rand(100, 4) * 20 + 1470
    y_train = 1.5 - 0.001 * X_train[:, 0] + 0.0005 * X_train[:, 1] + np.random.normal(0, 0.02, 100)
    
    X_test = np.random.rand(20, 4) * 20 + 1470
    y_test = 1.5 - 0.001 * X_test[:, 0] + 0.0005 * X_test[:, 1]
    
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Create ensemble
    print("\n[2] Building ensemble...")
    ensemble = FOSEnsemble()
    
    # Add models
    ensemble.add_model(create_model('linear'))
    ensemble.add_model(create_model('random_forest', n_estimators=50))
    ensemble.add_model(create_model('gradient_boost', n_estimators=50))
    
    try:
        ensemble.add_model(create_model('xgboost', n_estimators=50))
    except:
        print("  ⚠️  XGBoost not available, skipping")
    
    print(f"  ✓ Ensemble has {len(ensemble.models)} models")
    
    # Train ensemble
    print("\n[3] Training ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Test predictions
    print("\n[4] Testing predictions...")
    
    # Single prediction
    test_point = X_test[0]
    result = predict_with_confidence(ensemble, test_point)
    
    print(f"\n  Test point: {test_point}")
    print(f"  {result}")
    print(f"  Failure probability: {result.get_failure_probability(1.0):.1%}")
    print(f"  Risk of FOS < 1.1: {result.is_failure_risk(1.1)}")
    
    # Batch predictions
    mean, std, conf = ensemble.predict_with_uncertainty(X_test)
    
    print(f"\n  Batch test ({len(X_test)} samples):")
    print(f"    Mean FOS: {mean.mean():.3f}")
    print(f"    Avg uncertainty: {std.mean():.3f}")
    print(f"    Avg confidence: {conf.mean():.1%}")
    
    # Quantiles (95% CI)
    quantiles = ensemble.predict_quantiles(X_test, [0.025, 0.975])
    
    print(f"\n  95% Confidence Intervals:")
    for i in range(min(5, len(X_test))):
        print(f"    Sample {i+1}: [{quantiles[0.025][i]:.3f}, {quantiles[0.975][i]:.3f}]")
    
    # Evaluate on test set
    from sklearn.metrics import mean_squared_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_test, mean))
    r2 = r2_score(y_test, mean)
    
    print(f"\n[5] Ensemble Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Compare to individual models
    print(f"\n[6] Individual Model Performance:")
    for model in ensemble.models:
        y_pred_individual = model.predict(X_test)
        rmse_individual = np.sqrt(mean_squared_error(y_test, y_pred_individual))
        print(f"  {model.name:20s}: RMSE = {rmse_individual:.4f}")
    
    print("\n✅ Ensemble test complete!")
    print(f"\n💡 Key insight: Ensemble uncertainty (std) tells you when to trust predictions!")
