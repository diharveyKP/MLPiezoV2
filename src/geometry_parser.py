"""
Geometry Parser - Extract Dam Geometry from GeoStudio XML
"""

import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def extract_ground_surface(xml_path: str) -> np.ndarray:
    """
    Extract ground surface coordinates from GeoStudio XML.
    
    Args:
        xml_path: Path to GeoStudio XML file
        
    Returns:
        Array of (X, Y) coordinates defining ground surface
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find geometry
    geometry = root.find('.//Geometries/Geometry')
    
    if geometry is None:
        raise ValueError("No geometry found in XML")
    
    # Extract all points
    points_dict = {}
    points_elem = geometry.find('Points')
    
    if points_elem is not None:
        for point in points_elem.findall('Point'):
            point_id = point.get('ID')
            x = float(point.get('X', 0))
            y = float(point.get('Y', 0))
            points_dict[point_id] = (x, y)
    
    print(f"Found {len(points_dict)} geometry points")
    
    # Find ground surface line
    # Strategy: Find topmost continuous line
    ground_surface = find_topmost_line(points_dict, geometry)
    
    if ground_surface is None:
        # Fallback: Get envelope of all points
        all_points = np.array(list(points_dict.values()))
        ground_surface = create_envelope(all_points)
    
    # Sort by X
    ground_surface = ground_surface[ground_surface[:, 0].argsort()]
    
    return ground_surface


def find_topmost_line(points_dict: dict, geometry) -> Optional[np.ndarray]:
    """Find the topmost continuous line (ground surface)."""
    
    lines_elem = geometry.find('Lines')
    
    if lines_elem is None:
        return None
    
    # Build connectivity graph
    connections = {}
    
    for line in lines_elem.findall('Line'):
        p1_id = line.find('PointID1')
        p2_id = line.find('PointID2')
        
        if p1_id is not None and p2_id is not None:
            p1 = p1_id.text
            p2 = p2_id.text
            
            if p1 not in connections:
                connections[p1] = []
            if p2 not in connections:
                connections[p2] = []
            
            connections[p1].append(p2)
            connections[p2].append(p1)
    
    # Find longest connected path near top
    # Simple approach: Get points with Y in top 30%
    all_y = [pt[1] for pt in points_dict.values()]
    y_threshold = np.percentile(all_y, 70)
    
    top_points = []
    for pt_id, (x, y) in points_dict.items():
        if y >= y_threshold:
            top_points.append([x, y])
    
    if top_points:
        return np.array(top_points)
    
    return None


def create_envelope(points: np.ndarray) -> np.ndarray:
    """Create upper envelope of points."""
    
    # Bin by X, take max Y in each bin
    x_bins = np.linspace(points[:, 0].min(), points[:, 0].max(), 50)
    
    envelope = []
    
    for i in range(len(x_bins) - 1):
        mask = (points[:, 0] >= x_bins[i]) & (points[:, 0] < x_bins[i+1])
        if np.any(mask):
            x_avg = points[mask, 0].mean()
            y_max = points[mask, 1].max()
            envelope.append([x_avg, y_max])
    
    return np.array(envelope) if envelope else points


# ==================== TEST CODE ====================

if __name__ == "__main__":
    from tkinter import Tk, filedialog
    import matplotlib.pyplot as plt
    
    print("="*70)
    print("GEOMETRY PARSER TEST")
    print("="*70)
    
    # File dialog
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("\nSelect GeoStudio XML file...")
    
    xml_path = filedialog.askopenfilename(
        title='Select GeoStudio XML (SLOPE/W or SEEP/W)',
        filetypes=[('XML files', '*.xml'), ('All files', '*.*')],
        initialdir=str((PROJECT_ROOT / 'models').resolve())
    )
    
    root.destroy()
    
    if not xml_path:
        print("No file selected")
        sys.exit(0)
    
    print(f"✓ Selected: {Path(xml_path).name}")
    
    try:
        # Extract geometry
        ground = extract_ground_surface(xml_path)
        
        print(f"\n✅ Extracted ground surface:")
        print(f"  Points: {len(ground)}")
        print(f"  X range: [{ground[:, 0].min():.1f}, {ground[:, 0].max():.1f}]")
        print(f"  Y range: [{ground[:, 1].min():.1f}, {ground[:, 1].max():.1f}]")
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(ground[:, 0], ground[:, 1], 'k-', linewidth=3, label='Ground Surface')
        ax.fill_between(ground[:, 0], 0, ground[:, 1], alpha=0.3, color='saddlebrown')
        
        ax.set_xlabel('Distance (ft)', fontweight='bold', fontsize=13)
        ax.set_ylabel('Elevation (ft)', fontweight='bold', fontsize=13)
        ax.set_title('Extracted Ground Surface', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Export
        out_csv = Path(xml_path).with_suffix('.geometry.csv')
        import pandas as pd
        pd.DataFrame(ground, columns=['X', 'Y']).to_csv(out_csv, index=False)
        
        print(f"\n💾 Exported to: {out_csv}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
