"""
Dataset Manager - GENERALIZED VERSION
Handles any number of control points dynamically
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


class DatasetManager:
    """Dataset manager that adapts to any number of control points."""
    
    def __init__(self, output_dir: str, mode: str, version: str = "v1"):
        """
        Args:
            output_dir: Where to save dataset
            mode: 'shallow' or 'deep'
            version: Dataset version
        """
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.version = version
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_csv = self.output_dir / f"{mode}.csv"
        self.results_parquet = self.output_dir / f"{mode}.parquet"
        self.checkpoint_parquet = self.output_dir / f"checkpoint_{mode}.parquet"
        self.checkpoint_csv = self.output_dir / f"checkpoint_{mode}.csv"
        self.metadata_json = self.output_dir / f"{mode}_metadata.json"
        
        self.results = []
        self.n_total = 0
        self.n_success = 0
        self.n_failed = 0
        
        self._load_checkpoint_if_exists()
        logger.info(f"Dataset manager [{mode}]: {self.output_dir}, existing: {self.n_total}")
    
    def add_sample(self, sample_id: int, control_point_y: List[float], 
                   interpolated_y: np.ndarray, result: Any) -> None:
        """Add sample - DYNAMIC for any number of control points."""
        self.n_total += 1
        
        # Base record
        record = {
            'sample_id': sample_id,
            'mode': self.mode,
            'timestamp': datetime.now().isoformat(),
            'phreatic_surface_y': json.dumps(interpolated_y.tolist()),
            'success': result.success,
            'fos': result.fos if result.success else None,
            'solve_time': result.solve_time,
            'error_message': result.error_message
        }
        
        # Add control points DYNAMICALLY
        for i, y_value in enumerate(control_point_y, start=1):
            record[f'control_y_{i}'] = y_value
        
        # Update counters
        if result.success:
            self.n_success += 1
        else:
            self.n_failed += 1
        
        self.results.append(record)
    
    def save_checkpoint(self) -> None:
        """Save checkpoint (CSV for compatibility)."""
        if self.results:
            try:
                df = pd.DataFrame(self.results)
                # Use CSV for checkpoint (no pyarrow needed)
                df.to_csv(self.checkpoint_csv, index=False)
                logger.debug(f"Checkpoint: {len(self.results)} samples")
            except Exception as e:
                logger.error(f"Checkpoint failed: {e}")
    
    def save_final(self) -> None:
        """Save final dataset."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        try:
            df = pd.DataFrame(self.results)
            
            # CSV
            df.to_csv(self.results_csv, index=False)
            logger.info(f"Saved: {self.results_csv.name} ({len(df)} samples)")
            
            # Parquet
            try:
                df.to_parquet(self.results_parquet, index=False)
                logger.info(f"Saved: {self.results_parquet.name}")
            except:
                logger.warning("Parquet save failed (pyarrow not available)")
            
            # Metadata
            metadata = {
                'version': self.version,
                'mode': self.mode,
                'created': datetime.now().isoformat(),
                'n_total': self.n_total,
                'n_success': self.n_success,
                'n_failed': self.n_failed,
                'success_rate': round(self.n_success / self.n_total * 100, 2) if self.n_total > 0 else 0
            }
            
            with open(self.metadata_json, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self._save_summary()
            
        except Exception as e:
            logger.error(f"Save failed: {e}")
    
    def _save_summary(self) -> None:
        """Save text summary."""
        summary_path = self.output_dir / f"SUMMARY_{self.mode}.txt"
        
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"DATASET SUMMARY - {self.mode.upper()}\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total: {self.n_total}\n")
            f.write(f"Successful: {self.n_success}\n")
            f.write(f"Failed: {self.n_failed}\n")
            
            if self.n_success > 0:
                df = pd.DataFrame(self.results)
                df_success = df[df['success'] == True]
                fos = df_success['fos']
                
                f.write(f"\nFOS:\n")
                f.write(f"  Min: {fos.min():.4f}\n")
                f.write(f"  Max: {fos.max():.4f}\n")
                f.write(f"  Mean: {fos.mean():.4f}\n")
    
    def _load_checkpoint_if_exists(self) -> None:
        """Resume from checkpoint."""
        if self.results_csv.exists():
            try:
                df = pd.read_csv(self.results_csv)
                self.results = df.to_dict('records')
                self.n_total = len(self.results)
                self.n_success = sum(1 for r in self.results if r.get('success'))
                self.n_failed = self.n_total - self.n_success
                logger.info(f"Resumed from final dataset: {self.n_total} samples")
            except Exception as e:
                logger.error(f"Final dataset load failed: {e}")
        # Try checkpoint CSV next (more compatible)
        elif self.checkpoint_csv.exists():
            try:
                df = pd.read_csv(self.checkpoint_csv)
                self.results = df.to_dict('records')
                self.n_total = len(self.results)
                self.n_success = sum(1 for r in self.results if r.get('success'))
                self.n_failed = self.n_total - self.n_success
                logger.info(f"Resumed: {self.n_total} samples")
            except Exception as e:
                logger.error(f"Checkpoint load failed: {e}")
        elif self.checkpoint_parquet.exists():
            try:
                df = pd.read_parquet(self.checkpoint_parquet)
                self.results = df.to_dict('records')
                self.n_total = len(self.results)
                self.n_success = sum(1 for r in self.results if r.get('success'))
                self.n_failed = self.n_total - self.n_success
                logger.info(f"Resumed: {self.n_total} samples")
            except Exception as e:
                logger.error(f"Checkpoint load failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        stats = {
            'n_total': self.n_total,
            'n_success': self.n_success,
            'n_failed': self.n_failed,
            'success_rate': round(self.n_success / self.n_total * 100, 2) if self.n_total > 0 else 0
        }
        
        if self.n_success > 0:
            df = pd.DataFrame(self.results)
            df_success = df[df['success'] == True]
            fos = df_success['fos']
            
            stats['fos'] = {
                'min': float(fos.min()),
                'max': float(fos.max()),
                'mean': float(fos.mean()),
                'median': float(fos.median())
            }
        
        return stats


class ProgressTracker:
    """Progress tracker."""
    
    def __init__(self, total_samples: int, checkpoint_freq: int = 10):
        self.total_samples = total_samples
        self.checkpoint_freq = checkpoint_freq
        self.start_time = time.time()
    
    def print_progress(self, current: int, dm: DatasetManager) -> None:
        """Print progress."""
        elapsed = time.time() - self.start_time
        pct = current / self.total_samples * 100
        
        if current > 0:
            rate = current / elapsed
            eta = (self.total_samples - current) / rate if rate > 0 else 0
            eta_str = f"{eta/60:.1f}min" if eta < 3600 else f"{eta/3600:.1f}h"
        else:
            eta_str = "..."
        
        success_rate = dm.n_success / current * 100 if current > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"PROGRESS: {current}/{self.total_samples} ({pct:.1f}%)")
        print(f"Success: {dm.n_success} ({success_rate:.1f}%) | Failed: {dm.n_failed}")
        print(f"Elapsed: {elapsed/60:.1f}min | ETA: {eta_str}")
        
        if dm.n_success > 0:
            stats = dm.get_statistics()
            if 'fos' in stats:
                print(f"FOS: [{stats['fos']['min']:.3f}, {stats['fos']['max']:.3f}]")
    
    def should_checkpoint(self, current: int) -> bool:
        return current % self.checkpoint_freq == 0
