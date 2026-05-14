"""
GeoStudio interface for single-template solve workflows.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GeoStudioError(Exception):
    pass


class XMLUpdateError(GeoStudioError):
    pass


class PhysicsError(GeoStudioError):
    pass


class CSVReadError(GeoStudioError):
    pass


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
    """Result from a single GeoStudio run."""

    success: bool
    fos: Optional[float]
    solve_time: float
    error_message: Optional[str]
    csv_path: str


def _backup_xml(xml_path: str) -> bytes:
    return Path(xml_path).read_bytes()


def _restore_xml(xml_path: str, original_bytes: bytes) -> None:
    Path(xml_path).write_bytes(original_bytes)


def update_phreatic_datapoints(
    xml_path: str,
    datapoint_numbers: List[int],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> bool:
    """Update phreatic surface datapoints in XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        entry = root.find(".//StabilityItems/StabilityItem/Entry")
        if entry is None:
            raise XMLUpdateError("StabilityItem/Entry not found")

        datapoints_elem = entry.find("DataPoints")
        if datapoints_elem is None:
            raise XMLUpdateError("DataPoints element not found")

        datapoints_elem.clear()
        datapoints_elem.set("Len", str(len(datapoint_numbers)))

        for dp_num, x, y in zip(datapoint_numbers, x_coords, y_coords):
            dp = ET.SubElement(datapoints_elem, "DataPoint")
            dp.set("Number", str(dp_num))
            dp.set("X", f"{x:.5f}")
            dp.set("Y", f"{y:.2f}")

        piezo_surf = entry.find(".//PiezometricSurfaces/PiezometricSurface/DataPoints")
        if piezo_surf is not None:
            piezo_surf.clear()
            piezo_surf.set("Len", str(len(datapoint_numbers)))
            for dp_num in datapoint_numbers:
                ref = ET.SubElement(piezo_surf, "DataPoint")
                ref.text = str(dp_num)

        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        logger.info("Updated %s DataPoints", len(datapoint_numbers))
        return True
    except Exception as exc:
        logger.error("XML update failed: %s", exc)
        return False


def run_geocmd(xml_path: str, geocmd_exe: str, retries: int = 3, timeout: int = 300) -> bool:
    """Run GeoCMD with retries."""
    command = [geocmd_exe, str(xml_path), "/solve"]

    print(f"\n  [RUN] GeoCMD command: {os.path.basename(geocmd_exe)} {Path(xml_path).name}")
    print(f"     Template folder: {Path(xml_path).parent}")

    for attempt in range(1, retries + 1):
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
            logger.info("GeoCMD succeeded (attempt %s)", attempt)
            print("  [OK] GeoCMD completed successfully")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"\n  [FAIL] GeoCMD attempt {attempt} failed (exit code {exc.returncode})")
            if exc.stdout:
                print(f"     STDOUT: {exc.stdout[:500]}")
            if exc.stderr:
                print(f"     STDERR: {exc.stderr[:500]}")

            output = (exc.stdout or "") + (exc.stderr or "")
            if "convergence" in output.lower() or "physics" in output.lower():
                raise PhysicsError("Physics/convergence error")

            if attempt < retries:
                time.sleep(2)
            else:
                return False
        except subprocess.TimeoutExpired:
            print(f"  [FAIL] GeoCMD timed out (attempt {attempt})")
            if attempt >= retries:
                return False
            time.sleep(2)

    return False


def read_fos_from_csv(
    csv_path: str,
    column_name: str = "SlipFOS",
    timeout: int = 180,
    min_mtime_ns: int | None = None,
) -> Optional[float]:
    """Read FOS from CSV, waiting for a fresh enough file to appear."""
    deadline = time.time() + timeout
    csv_path_obj = Path(csv_path)

    while time.time() < deadline:
        try:
            if csv_path_obj.exists() and csv_path_obj.stat().st_size > 256:
                if min_mtime_ns is not None and csv_path_obj.stat().st_mtime_ns < min_mtime_ns:
                    time.sleep(0.5)
                    continue

                df = pd.read_csv(csv_path_obj)
                if column_name not in df.columns:
                    raise CSVReadError(f"Column '{column_name}' not found")

                fos_series = pd.to_numeric(df[column_name], errors="coerce")
                min_fos = fos_series.min()
                if pd.isna(min_fos):
                    raise CSVReadError("FOS is NaN")

                logger.info("Read FOS = %.4f", min_fos)
                return round(float(min_fos), 4)
        except Exception:
            pass

        time.sleep(0.5)

    raise CSVReadError(f"Timeout waiting for CSV: {csv_path}")


class GeoStudioInterface:
    """
    Single-template GeoStudio interface.
    """

    def __init__(self, template_path: str, geocmd_exe: str, fos_column: str = "SlipFOS"):
        self.template_path = Path(template_path)
        self.template_folder = self.template_path.parent
        self.geocmd_exe = geocmd_exe
        self.fos_column = fos_column

        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        if not os.path.exists(geocmd_exe):
            raise FileNotFoundError(f"GeoCMD not found: {geocmd_exe}")

        logger.info("GeoStudio interface initialized")
        logger.info("  Template: %s", self.template_path)
        logger.info("  Working in: %s", self.template_folder)

    def _result_csv_path(self) -> Path:
        return self.template_folder / "Analysis" / "001" / "slip_surface.csv"

    def _clear_stale_results(self) -> None:
        csv_path = self._result_csv_path()
        if csv_path.exists():
            csv_path.unlink()

    def run_analysis(self, phreatic_data: PhreaticSurfaceData, sample_id: int) -> GeoStudioResult:
        """
        Run a single analysis using an updated temporary template state.
        """
        del sample_id
        start_time = time.perf_counter()
        template_backup = _backup_xml(str(self.template_path))
        csv_path = self._result_csv_path()

        print("\n  [STEP] Updating phreatic surface in template...")

        try:
            self._clear_stale_results()

            datapoint_numbers = list(range(1, len(phreatic_data.interpolated_y) + 1))
            success = update_phreatic_datapoints(
                xml_path=str(self.template_path),
                datapoint_numbers=datapoint_numbers,
                x_coords=phreatic_data.interpolated_x,
                y_coords=phreatic_data.interpolated_y,
            )
            if not success:
                return GeoStudioResult(False, None, 0.0, "XML update failed", "")

            print(f"  [OK] Updated {len(datapoint_numbers)} DataPoints")

            run_start_ns = time.time_ns()
            success = run_geocmd(
                xml_path=str(self.template_path),
                geocmd_exe=self.geocmd_exe,
                retries=3,
                timeout=300,
            )
            if not success:
                return GeoStudioResult(False, None, time.perf_counter() - start_time, "GeoCMD failed", "")

            print(f"  [STEP] Reading FOS from {csv_path.name}...")
            fos = read_fos_from_csv(str(csv_path), self.fos_column, timeout=180, min_mtime_ns=run_start_ns)
            if fos is None:
                return GeoStudioResult(False, None, time.perf_counter() - start_time, "Failed to read FOS", str(csv_path))

            solve_time = time.perf_counter() - start_time
            print(f"  [OK] FOS = {fos:.4f} ({solve_time:.1f}s)")
            return GeoStudioResult(True, fos, solve_time, None, str(csv_path))
        except PhysicsError as exc:
            return GeoStudioResult(False, None, time.perf_counter() - start_time, f"Physics error: {exc}", "")
        except Exception as exc:
            return GeoStudioResult(False, None, time.perf_counter() - start_time, f"GeoStudio error: {exc}", "")
        finally:
            _restore_xml(str(self.template_path), template_backup)
