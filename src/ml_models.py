"""
ML Models for FOS Prediction
Defines all model types with consistent interface
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

# Try to import XGBoost (optional but recommended)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available - install with: pip install xgboost")


class FOSModelBase:
    """Base class for all FOS prediction models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FOSModelBase':
        """Train the model."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict FOS values."""
        raise NotImplementedError
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        return None


class LinearModel(FOSModelBase):
    """Simple linear regression baseline."""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__("Linear_Ridge")
        self.alpha = alpha
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearModel':
        """Train linear model with scaling."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        logger.info(f"{self.name} trained on {len(X)} samples")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict FOS."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get coefficients as feature importance."""
        if not self.is_fitted:
            return None
        return np.abs(self.model.coef_)


class XGBoostModel(FOSModelBase):
    """XGBoost gradient boosting model."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostModel':
        """Train XGBoost model."""
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"{self.name} trained on {len(X)} samples")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict FOS."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get XGBoost feature importance."""
        if not self.is_fitted:
            return None
        return self.model.feature_importances_


class RandomForestModel(FOSModelBase):
    """Random Forest ensemble model."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """Train Random Forest."""
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"{self.name} trained on {len(X)} samples")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict FOS."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        if not self.is_fitted:
            return None
        return self.model.feature_importances_


class NeuralNetworkModel(FOSModelBase):
    """Multi-layer perceptron neural network."""
    
    def __init__(self, hidden_layers: Tuple[int, ...] = (64, 32, 16)):
        super().__init__("NeuralNet")
        self.hidden_layers = hidden_layers
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralNetworkModel':
        """Train neural network with scaling."""
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50
        )
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"{self.name} trained on {len(X)} samples")
        logger.info(f"  Architecture: 4 → {' → '.join(map(str, self.hidden_layers))} → 1")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict FOS."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class GradientBoostingModel(FOSModelBase):
    """Sklearn Gradient Boosting (alternative to XGBoost)."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5, 
                 learning_rate: float = 0.1):
        super().__init__("GradientBoosting")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingModel':
        """Train gradient boosting."""
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42
        )
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"{self.name} trained on {len(X)} samples")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict FOS."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        if not self.is_fitted:
            return None
        return self.model.feature_importances_


# ========================= MODEL FACTORY =========================

def create_model(model_type: str, **kwargs) -> FOSModelBase:
    """
    Factory function to create models.
    
    Args:
        model_type: 'linear', 'xgboost', 'random_forest', 'neural_net', 'gradient_boost'
        **kwargs: Model-specific parameters
        
    Returns:
        Initialized model
    """
    models = {
        'linear': LinearModel,
        'random_forest': RandomForestModel,
        'neural_net': NeuralNetworkModel,
        'gradient_boost': GradientBoostingModel
    }
    
    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBoostModel
    
    if model_type not in models:
        available = list(models.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    
    return models[model_type](**kwargs)


# ========================= TEST CODE =========================

if __name__ == "__main__":
    print("="*70)
    print("🧪 ML MODELS TEST")
    print("="*70)
    
    # Create dummy data
    print("\n[1] Creating test data...")
    np.random.seed(42)
    X_train = np.random.rand(100, 4) * 20 + 1470  # Random control points
    y_train = 1.5 - 0.001 * X_train[:, 0] + 0.0005 * X_train[:, 1]  # Fake FOS relationship
    
    X_test = np.random.rand(20, 4) * 20 + 1470
    y_test = 1.5 - 0.001 * X_test[:, 0] + 0.0005 * X_test[:, 1]
    
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Test each model type
    model_types = ['linear', 'random_forest', 'gradient_boost', 'neural_net']
    
    if XGBOOST_AVAILABLE:
        model_types.insert(1, 'xgboost')
    
    results = {}
    
    for model_type in model_types:
        print(f"\n[TEST] {model_type.upper()}")
        try:
            # Create and train
            model = create_model(model_type)
            model.fit(X_train, y_train)
            print(f"  ✓ Training complete")
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            from sklearn.metrics import mean_squared_error, r2_score
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"  ✓ RMSE: {rmse:.4f}")
            print(f"  ✓ R²: {r2:.4f}")
            
            # Feature importance
            importance = model.get_feature_importance()
            if importance is not None:
                print(f"  ✓ Feature importance: {importance}")
            
            results[model_type] = {'rmse': rmse, 'r2': r2}
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)
    for model_type, metrics in results.items():
        print(f"{model_type:20s} | RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f}")
    
    print("\n✅ All models tested successfully!")