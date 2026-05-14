"""
GeoStudio Interface - SIMPLIFIED Single Mode
Runs analysis directly in template folder
"""

import os
import time
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ========================= EXCEPTIONS =========================

class GeoStudioError(Exception):
    pass

class XMLUpdateError(GeoStudioError):
    pass

class PhysicsError(GeoStudioError):
    pass

class CSVReadError(GeoStudioError):
    pass


# ========================= DATA STRUCTURES =========================

@dataclass
class PhreaticSurfaceData:
    """Phreatic surface data."""
    control_point_numbers: List[int]
    control_point_x: List[float]
    control_point_y: List[float]
    interpolated_x: np.ndarray
    interpolated_y: np.ndarray


@dataclass
class GeoStudioResult:
    """Result from single GeoStudio run."""
    success: bool
    fos: Optional[float]
    solve_time: float
    error_message: Optional[str]
    csv_path: str


# ========================= XML UPDATE =========================

@contextmanager
def safe_xml_update(xml_path: str):
    """Context manager for safe XML updates with automatic backup/restore."""
    xml_path_obj = Path(xml_path)
    backup_path = xml_path_obj.with_suffix('.xml.backup')
    
    try:
        # Create backup
        backup_path.write_bytes(xml_path_obj.read_bytes())
        
        # Parse and yield for modification
        tree = ET.parse(xml_path)
        yield tree
        
        # Write modified XML
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        
    except Exception as e:
        # Restore backup on any error
        if backup_path.exists():
            xml_path_obj.write_bytes(backup_path.read_bytes())
        raise XMLUpdateError(f"XML update failed: {e}") from e
    
    finally:
        # Clean up backup
        if backup_path.exists():
            backup_path.unlink()


def update_phreatic_datapoints(xml_path: str, datapoint_numbers: List[int],
                               x_coords: np.ndarray, y_coords: np.ndarray) -> bool:
    """Update phreatic surface DataPoints in XML."""
    try:
        with safe_xml_update(xml_path) as tree:
            root = tree.getroot()
            
            # Find DataPoints in StabilityItems
            entry = root.find('.//StabilityItems/StabilityItem/Entry')
            if entry is None:
                raise XMLUpdateError("StabilityItem/Entry not found")
            
            datapoints_elem = entry.find('DataPoints')
            if datapoints_elem is None:
                raise XMLUpdateError("DataPoints element not found")
            
            # Clear and rebuild
            datapoints_elem.clear()
            datapoints_elem.set('Len', str(len(datapoint_numbers)))
            
            for dp_num, x, y in zip(datapoint_numbers, x_coords, y_coords):
                dp = ET.SubElement(datapoints_elem, 'DataPoint')
                dp.set('Number', str(dp_num))
                dp.set('X', f"{x:.5f}")
                dp.set('Y', f"{y:.2f}")
            
            # Update PiezometricSurface references
            piezo_surf = entry.find('.//PiezometricSurfaces/PiezometricSurface/DataPoints')
            if piezo_surf is not None:
                piezo_surf.clear()
                piezo_surf.set('Len', str(len(datapoint_numbers)))
                for dp_num in datapoint_numbers:
                    ref = ET.SubElement(piezo_surf, 'DataPoint')
                    ref.text = str(dp_num)
            
            logger.info(f"Updated {len(datapoint_numbers)} DataPoints")
            return True
            
    except Exception as e:
        logger.error(f"XML update failed: {e}")
        return False


# ========================= GEOCMD RUNNER =========================

def run_geocmd(xml_path: str, geocmd_exe: str, retries: int = 3, timeout: int = 300) -> bool:
    """Run GeoCMD with retries."""
    command = [geocmd_exe, str(xml_path), "/solve"]
    
    print(f"\n  🔧 GeoCMD command: {os.path.basename(geocmd_exe)} {Path(xml_path).name}")
    print(f"     Template folder: {Path(xml_path).parent}")
    
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(command, check=True, capture_output=True, 
                                  text=True, timeout=timeout)
            logger.info(f"GeoCMD succeeded (attempt {attempt})")
            print(f"  ✓ GeoCMD completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n  ❌ GeoCMD attempt {attempt} failed (exit code {e.returncode})")
            
            if e.stdout:
                print(f"     STDOUT: {e.stdout[:500]}")
            if e.stderr:
                print(f"     STDERR: {e.stderr[:500]}")
            
            # Check for physics errors
            output = (e.stdout or "") + (e.stderr or "")
            if "convergence" in output.lower() or "physics" in output.lower():
                raise PhysicsError(f"Physics/convergence error")
            
            if attempt < retries:
                time.sleep(2)
            else:
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ❌ GeoCMD timed out (attempt {attempt})")
            if attempt >= retries:
                return False
            time.sleep(2)
    
    return False


# ========================= CSV READER =========================

def read_fos_from_csv(csv_path: str, column_name: str = "SlipFOS", 
                     timeout: int = 180) -> Optional[float]:
    """Read FOS from CSV, waiting for file to appear."""
    deadline = time.time() + timeout
    
    while time.time() < deadline:
        try:
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 256:
                df = pd.read_csv(csv_path)
                
                if column_name not in df.columns:
                    raise CSVReadError(f"Column '{column_name}' not found")
                
                fos_series = pd.to_numeric(df[column_name], errors='coerce')
                min_fos = fos_series.min()
                
                if pd.isna(min_fos):
                    raise CSVReadError("FOS is NaN")
                
                logger.info(f"Read FOS = {min_fos:.4f}")
                return round(float(min_fos), 4)
        except:
            pass
        
        time.sleep(0.5)
    
    raise CSVReadError(f"Timeout waiting for CSV: {csv_path}")


# ========================= MAIN INTERFACE =========================

class GeoStudioInterface:
    """
    SIMPLIFIED GeoStudio interface - single mode operation.
    Runs analysis directly in template folder.
    """
    
    def __init__(self, template_path: str, geocmd_exe: str, fos_column: str = "SlipFOS"):
        """
        Args:
            template_path: Full path to template XML file
            geocmd_exe: Full path to geocmd.exe
            fos_column: CSV column name for FOS
        """
        self.template_path = Path(template_path)
        self.template_folder = self.template_path.parent
        self.geocmd_exe = geocmd_exe
        self.fos_column = fos_column
        
        # Validate
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        if not os.path.exists(geocmd_exe):
            raise FileNotFoundError(f"GeoCMD not found: {geocmd_exe}")
        
        logger.info(f"GeoStudio interface initialized")
        logger.info(f"  Template: {self.template_path}")
        logger.info(f"  Working in: {self.template_folder}")
    
    def run_analysis(self, phreatic_data: PhreaticSurfaceData, sample_id: int) -> GeoStudioResult:
        """
        Run single analysis.
        
        Process:
        1. Update template XML with new phreatic surface
        2. Run GeoCMD
        3. Read FOS from CSV
        4. XML auto-restores from backup
        """
        start_time = time.perf_counter()
        
        print(f"\n  📝 Updating phreatic surface in template...")
        
        # UPDATE XML
        try:
            datapoint_numbers = list(range(1, len(phreatic_data.interpolated_y) + 1))
            
            success = update_phreatic_datapoints(
                xml_path=str(self.template_path),
                datapoint_numbers=datapoint_numbers,
                x_coords=phreatic_data.interpolated_x,
                y_coords=phreatic_data.interpolated_y
            )
            
            if not success:
                return GeoStudioResult(False, None, 0.0, "XML update failed", "")
            
            print(f"  ✓ Updated {len(datapoint_numbers)} DataPoints")
            
        except Exception as e:
            return GeoStudioResult(False, None, 0.0, f"XML error: {e}", "")
        
        # RUN GEOSTUDIO
        try:
            success = run_geocmd(
                xml_path=str(self.template_path),
                geocmd_exe=self.geocmd_exe,
                retries=3,
                timeout=300
            )
            
            if not success:
                return GeoStudioResult(False, None, time.perf_counter() - start_time, 
                                     "GeoCMD failed", "")
                
        except PhysicsError as e:
            return GeoStudioResult(False, None, time.perf_counter() - start_time, 
                                 f"Physics error: {e}", "")
        except Exception as e:
            return GeoStudioResult(False, None, time.perf_counter() - start_time, 
                                 f"GeoStudio error: {e}", "")
        
        # READ RESULTS
        try:
            csv_path = self.template_folder / "Analysis" / "001" / "slip_surface.csv"
            
            print(f"  📊 Reading FOS from {csv_path.name}...")
            
            fos = read_fos_from_csv(str(csv_path), self.fos_column, timeout=180)
            
            if fos is None:
                return GeoStudioResult(False, None, time.perf_counter() - start_time, 
                                     "Failed to read FOS", str(csv_path))
            
            solve_time = time.perf_counter() - start_time
            
            print(f"  ✅ FOS = {fos:.4f} ({solve_time:.1f}s)")
            
            return GeoStudioResult(True, fos, solve_time, None, str(csv_path))
            
        except Exception as e:
            return GeoStudioResult(False, None, time.perf_counter() - start_time, 
                                 f"CSV error: {e}", "")
