"""
SEEP/W XML Parser
Extract phreatic lines and hydraulic data from GeoStudio SEEP/W models
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


class SeepWParser:
    """
    Parse SEEP/W XML files to extract phreatic surfaces and hydraulic data.
    
    GeoStudio SEEP/W stores results in XML format.
    This parser extracts:
    - Phreatic surface coordinates
    - Pressure head contours  
    - Hydraulic gradient fields
    - Flow vectors
    """
    
    def __init__(self, xml_path: str):
        """
        Initialize parser.
        
        Args:
            xml_path: Path to SEEP/W .xml file
        """
        self.xml_path = Path(xml_path)
        
        if not self.xml_path.exists():
            raise FileNotFoundError(f"SEEP/W file not found: {xml_path}")
        
        logger.info(f"Loading SEEP/W model: {self.xml_path.name}")
        
        try:
            self.tree = ET.parse(self.xml_path)
            self.root = self.tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML: {e}")
        
        logger.info("✓ XML parsed successfully")
    
    def get_phreatic_surface(self, analysis_name: str = None) -> np.ndarray:
        """
        Extract phreatic surface coordinates.
        
        Args:
            analysis_name: Name of analysis (if multiple), None = first
            
        Returns:
            Array of shape (n_points, 2) with [X, Y] coordinates
        """
        logger.info("Extracting phreatic surface...")
        
        # Method 1: Try to find explicit phreatic line
        phreatic_line = self._find_phreatic_line_explicit()
        
        if phreatic_line is not None:
            logger.info(f"✓ Found phreatic line: {len(phreatic_line)} points")
            return phreatic_line
        
        # Method 2: Extract from pressure head = 0 contour
        logger.info("Trying pressure head = 0 contour...")
        phreatic_line = self._extract_from_pressure_contour()
        
        if phreatic_line is not None:
            logger.info(f"✓ Extracted from pressure contour: {len(phreatic_line)} points")
            return phreatic_line
        
        # Method 3: Parse from DataPoints
        logger.info("Trying DataPoints extraction...")
        phreatic_line = self._extract_from_datapoints()
        
        if phreatic_line is not None:
            logger.info(f"✓ Extracted from DataPoints: {len(phreatic_line)} points")
            return phreatic_line
        
        raise ValueError("Could not find phreatic surface in XML")
    
    def _find_phreatic_line_explicit(self) -> Optional[np.ndarray]:
        """Try to find explicitly defined phreatic line."""
        
        # Look for PiezometricSurface or similar
        for elem in self.root.iter():
            if 'Piezometric' in elem.tag or 'Phreatic' in elem.tag:
                # Found phreatic element
                datapoints = elem.find('.//DataPoints')
                
                if datapoints is not None:
                    points = []
                    
                    for dp in datapoints.findall('DataPoint'):
                        try:
                            x = float(dp.get('X', 0))
                            y = float(dp.get('Y', 0))
                            points.append([x, y])
                        except (ValueError, TypeError):
                            continue
                    
                    if points:
                        return np.array(points)
        
        return None
    
    def _extract_from_pressure_contour(self) -> Optional[np.ndarray]:
        """Extract phreatic line from pressure head = 0 contour."""
        
        # Look for contour data or function defining pressure = 0
        # This is model-specific and may need adjustment
        
        # Placeholder for now
        return None
    
    def _extract_from_datapoints(self) -> Optional[np.ndarray]:
        """Extract from generic DataPoints elements."""
        
        # Find all DataPoints elements
        all_datapoints = self.root.findall('.//DataPoints')
        
        for dp_elem in all_datapoints:
            points = []
            
            for dp in dp_elem.findall('DataPoint'):
                try:
                    x = float(dp.get('X', dp.get('x', 0)))
                    y = float(dp.get('Y', dp.get('y', 0)))
                    
                    # Filter out obviously wrong points (like Z coordinates)
                    if y > 100:  # Reasonable elevation check
                        points.append([x, y])
                        
                except (ValueError, TypeError, AttributeError):
                    continue
            
            # Return if we found a reasonable number of points
            if len(points) > 3:
                # Sort by X
                points_sorted = sorted(points, key=lambda p: p[0])
                return np.array(points_sorted)
        
        return None
    
    def get_pressure_heads(self) -> Optional[np.ndarray]:
        """
        Extract pressure head field.
        
        Returns:
            Array with (X, Y, pressure_head) for each point
        """
        # This would parse nodal results
        # Implementation depends on SEEP/W output format
        pass
    
    def get_hydraulic_gradients(self) -> Optional[np.ndarray]:
        """
        Extract hydraulic gradient field.
        
        Returns:
            Array with (X, Y, gradient_x, gradient_y) for each element
        """
        pass
    
    def export_to_csv(self, output_path: str, analysis_name: str = None) -> None:
        """
        Export phreatic surface to CSV.
        
        Args:
            output_path: Where to save CSV
            analysis_name: Which analysis to export
        """
        phreatic = self.get_phreatic_surface(analysis_name)
        
        df = pd.DataFrame(phreatic, columns=['X', 'Y'])
        df.to_csv(output_path, index=False)
        
        logger.info(f"✓ Exported to: {output_path}")
    
    def get_model_info(self) -> Dict:
        """Get general model information."""
        
        info = {}
        
        # File info
        file_info = self.root.find('.//FileInfo')
        if file_info is not None:
            info['title'] = file_info.get('Title', 'Unknown')
            info['author'] = file_info.get('Author', 'Unknown')
            info['date'] = file_info.get('Date', 'Unknown')
        
        # Analyses
        analyses = self.root.findall('.//Analysis')
        info['n_analyses'] = len(analyses)
        info['analysis_names'] = [a.find('Name').text for a in analyses if a.find('Name') is not None]
        
        return info


# ==================== BATCH PROCESSING ====================

class SeepWLibrary:
    """
    Manage library of SEEP/W phreatic surfaces.
    
    Handles multiple SEEP/W runs for different scenarios.
    """
    
    def __init__(self, library_dir: str):
        """
        Args:
            library_dir: Directory containing SEEP/W XML files
        """
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        
        self.surfaces = {}
    
    def add_from_xml(self, xml_path: str, scenario_name: str) -> None:
        """Add phreatic surface from SEEP/W XML."""
        
        parser = SeepWParser(xml_path)
        phreatic = parser.get_phreatic_surface()
        
        self.surfaces[scenario_name] = phreatic
        
        logger.info(f"Added scenario: {scenario_name} ({len(phreatic)} points)")
    
    def add_from_csv(self, csv_path: str, scenario_name: str) -> None:
        """Add phreatic surface from CSV."""
        
        df = pd.read_csv(csv_path)
        phreatic = df[['X', 'Y']].values
        
        self.surfaces[scenario_name] = phreatic
        
        logger.info(f"Added scenario: {scenario_name} ({len(phreatic)} points)")
    
    def get_surface(self, scenario_name: str) -> np.ndarray:
        """Get phreatic surface for scenario."""
        
        if scenario_name not in self.surfaces:
            raise KeyError(f"Scenario not found: {scenario_name}")
        
        return self.surfaces[scenario_name]
    
    def get_all_scenarios(self) -> List[str]:
        """Get list of all scenario names."""
        return list(self.surfaces.keys())
    
    def export_all(self) -> None:
        """Export all surfaces to CSV files."""
        
        for name, surface in self.surfaces.items():
            output_path = self.library_dir / f"{name}.csv"
            
            df = pd.DataFrame(surface, columns=['X', 'Y'])
            df.to_csv(output_path, index=False)
            
            logger.info(f"✓ Exported: {output_path.name}")
    
    def get_envelope(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get upper and lower envelope of all surfaces.
        
        Returns:
            (upper_envelope, lower_envelope) as (X, Y) arrays
        """
        if not self.surfaces:
            raise ValueError("No surfaces in library")
        
        # Get common X coordinates
        all_x = []
        for surf in self.surfaces.values():
            all_x.extend(surf[:, 0])
        
        x_common = np.linspace(min(all_x), max(all_x), 100)
        
        # Interpolate all surfaces to common X
        all_y = []
        for surf in self.surfaces.values():
            y_interp = np.interp(x_common, surf[:, 0], surf[:, 1])
            all_y.append(y_interp)
        
        all_y = np.array(all_y)
        
        upper = np.max(all_y, axis=0)
        lower = np.min(all_y, axis=0)
        
        return np.column_stack([x_common, upper]), np.column_stack([x_common, lower])


# ==================== TEST CODE ====================

if __name__ == "__main__":
    from tkinter import Tk, filedialog
    
    print("="*70)
    print("SEEP/W PARSER")
    print("="*70)
    
    # Ask user what to do
    print("\nOptions:")
    print("  [1] Parse SEEP/W XML file")
    print("  [2] Create dummy library for testing")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == '1':
        # File dialog to select SEEP/W XML
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        print("\nSelect SEEP/W XML file...")
        
        xml_path = filedialog.askopenfilename(
            title='Select SEEP/W XML File',
            filetypes=[
                ('GeoStudio XML', '*.xml'),
                ('All files', '*.*')
            ],
            initialdir=str((PROJECT_ROOT / 'models').resolve())
        )
        
        root.destroy()
        
        if not xml_path:
            print("❌ No file selected")
            sys.exit(0)
        
        print(f"✓ Selected: {Path(xml_path).name}")
        
        # Parse
        try:
            parser = SeepWParser(xml_path)
            
            # Model info
            info = parser.get_model_info()
            print(f"\nModel Info:")
            print(f"  Title: {info.get('title', 'N/A')}")
            print(f"  Author: {info.get('author', 'N/A')}")
            print(f"  Analyses: {info.get('n_analyses', 0)}")
            
            if info.get('analysis_names'):
                print(f"  Analysis names: {', '.join(info['analysis_names'])}")
            
            # Extract phreatic
            print(f"\nExtracting phreatic surface...")
            phreatic = parser.get_phreatic_surface()
            
            print(f"\n✅ SUCCESS!")
            print(f"  Points: {len(phreatic)}")
            print(f"  X range: [{phreatic[:, 0].min():.2f}, {phreatic[:, 0].max():.2f}] ft")
            print(f"  Y range: [{phreatic[:, 1].min():.2f}, {phreatic[:, 1].max():.2f}] ft")
            
            # Show first/last points
            print(f"\n  First point: X={phreatic[0, 0]:.2f}, Y={phreatic[0, 1]:.2f}")
            print(f"  Last point:  X={phreatic[-1, 0]:.2f}, Y={phreatic[-1, 1]:.2f}")
            
            # Export
            output_csv = Path(xml_path).with_suffix('.phreatic.csv')
            parser.export_to_csv(output_csv)
            
            print(f"\n💾 Exported to: {output_csv}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == '2':
        # Create dummy library
        print("\nCreating dummy SEEP/W library...")
        
        lib = SeepWLibrary('data/seepw_library')
        
        pond_levels = [3070, 3075, 3080, 3085, 3090, 3092]
        
        for pond in pond_levels:
            x = np.linspace(0, 1851.6, 100)
            
            # Simulate realistic phreatic surface
            # Linear drop + sinusoidal variation
            y = pond - (x / 1851.6) * (pond - 2895)
            y = y - 5 * np.sin(x / 400)  # Add realistic undulation
            
            # Ensure monotonic decrease
            for i in range(1, len(y)):
                if y[i] > y[i-1]:
                    y[i] = y[i-1]
            
            scenario = f"pond_{pond}"
            lib.surfaces[scenario] = np.column_stack([x, y])
            
            print(f"  ✓ Created: {scenario}")
        
        # Get envelope
        upper, lower = lib.get_envelope()
        
        print(f"\n✓ Library: {len(lib.surfaces)} scenarios")
        print(f"  Upper: Y=[{upper[:, 1].min():.1f}, {upper[:, 1].max():.1f}]")
        print(f"  Lower: Y=[{lower[:, 1].min():.1f}, {lower[:, 1].max():.1f}]")
        
        # Export
        lib.export_all()
        
        print(f"\n💾 Exported to: {lib.library_dir}")
        print("\n✅ Dummy library created!")
    
    else:
        print("Invalid option")
