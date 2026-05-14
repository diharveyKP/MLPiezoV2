"""
Phreatic Surface Generator
Interpolates 4 control points into 50-point smooth phreatic surface
-
"""

import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class PhreaticSurfaceGenerator:
    """
    Generates phreatic surfaces by interpolating control points.
    """
    
    def __init__(self, control_point_x: List[float], n_total_points: int = 50,
                 interpolation_method: str = "pchip", enforce_flow_direction: bool = True):
        self.control_point_x = np.array(control_point_x)
        self.n_control_points = len(control_point_x)
        self.n_total_points = n_total_points
        self.interpolation_method = interpolation_method
        self.enforce_flow_direction = enforce_flow_direction
        
        self.x_min = np.min(self.control_point_x)
        self.x_max = np.max(self.control_point_x)
        self.x_interpolated = np.linspace(self.x_min, self.x_max, n_total_points)
        
        logger.info(f"Phreatic generator initialized: {self.n_control_points} control points, {self.n_total_points} interpolated points")
    
    def generate(self, control_point_y: List[float]) -> np.ndarray:
        if len(control_point_y) != self.n_control_points:
            raise ValueError(f"Expected {self.n_control_points} control Y values, got {len(control_point_y)}")
        
        control_point_y = np.array(control_point_y)
        y_interpolated = self._interpolate(self.control_point_x, control_point_y)
        y_constrained = self._apply_constraints(y_interpolated)
        y_final = np.round(y_constrained, 2)
        return y_final
    
    def _interpolate(self, x_control: np.ndarray, y_control: np.ndarray) -> np.ndarray:
        if self.interpolation_method == "pchip":
            interpolator = PchipInterpolator(x_control, y_control)
        elif self.interpolation_method == "cubic":
            interpolator = CubicSpline(x_control, y_control, bc_type='natural')
        else:
            interpolator = lambda x: np.interp(x, x_control, y_control)
        
        return interpolator(self.x_interpolated)
    
    def _apply_constraints(self, y_values: np.ndarray) -> np.ndarray:
        y_constrained = y_values.copy()
        if self.enforce_flow_direction:
            for i in range(1, len(y_constrained)):
                if y_constrained[i] > y_constrained[i-1] + 0.01:
                    y_constrained[i] = y_constrained[i-1]
        return y_constrained
    
    def get_x_coordinates(self) -> np.ndarray:
        return self.x_interpolated.copy()


class PhreaticSurfaceSampler:
    """Generates samples of control point Y values."""
    
    def __init__(self, y_mins: List[float], y_maxs: List[float], y_baselines: List[float]):
        self.n_control_points = len(y_mins)
        self.y_mins = np.array(y_mins)
        self.y_maxs = np.array(y_maxs)
        self.y_baselines = np.array(y_baselines)
    
    def sample_sobol(self, n_samples: int, seed: int = 42) -> np.ndarray:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=self.n_control_points, scramble=True, seed=seed)
        unit_samples = sampler.random(n=n_samples)
        
        samples = np.zeros_like(unit_samples)
        for i in range(self.n_control_points):
            samples[:, i] = self.y_mins[i] + unit_samples[:, i] * (self.y_maxs[i] - self.y_mins[i])
        
        return np.round(samples, 2)
    
    def sample_lhs(self, n_samples: int, seed: int = 42) -> np.ndarray:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=self.n_control_points, seed=seed)
        unit_samples = sampler.random(n=n_samples)
        
        samples = np.zeros_like(unit_samples)
        for i in range(self.n_control_points):
            samples[:, i] = self.y_mins[i] + unit_samples[:, i] * (self.y_maxs[i] - self.y_mins[i])
        
        return np.round(samples, 2)
    
    def sample_random(self, n_samples: int, seed: int = 42) -> np.ndarray:
        np.random.seed(seed)
        samples = np.zeros((n_samples, self.n_control_points))
        for i in range(self.n_control_points):
            samples[:, i] = np.random.uniform(self.y_mins[i], self.y_maxs[i], size=n_samples)
        return np.round(samples, 2)