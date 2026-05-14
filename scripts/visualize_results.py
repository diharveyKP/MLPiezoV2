"""
FOS Dataset Visualization - 4 Safety Categories
<1.0 (Failure), 1.0-1.3 (Critical), 1.3-1.5 (Moderate), ≥1.5 (Safe)
"""

import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path
import xml.etree.ElementTree as ET

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.dpi': 100
})


def load_runtime_config(config_path: str | None) -> dict:
    if not config_path:
        return {}

    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")

    spec = importlib.util.spec_from_file_location("runtime_model_config", config_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {config_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"Loaded config: {config_file}")
    return {
        "control_points": module.CONTROL_POINTS,
        "dataset": module.DATASET,
    }


def _resolve_template_xml(config_path: str | None) -> Path | None:
    if not config_path:
        return None

    config_file = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("runtime_model_config_for_geometry", config_file)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    shallow = getattr(module, "GEOSTUDIO", {}).get("templates", {}).get("shallow")
    if shallow:
        xml_path = Path(shallow)
        if xml_path.exists():
            return xml_path

    workspace_root = config_file.parent
    fallback = workspace_root / "template_model"
    matches = sorted(fallback.glob("*.xml"))
    return matches[0] if matches else None


def load_geometry_profile(config_path: str | None, n_samples: int = 1200) -> np.ndarray | None:
    xml_path = _resolve_template_xml(config_path)
    if xml_path is None or not xml_path.exists():
        return None

    try:
        root = ET.parse(xml_path).getroot()
        geometry = root.find(".//Geometry")
        if geometry is None:
            return None

        point_map = {}
        for point in geometry.findall("./Points/Point"):
            point_id = point.attrib.get("ID")
            x_val = point.attrib.get("X")
            y_val = point.attrib.get("Y")
            if point_id is None or x_val is None or y_val is None:
                continue
            point_map[point_id] = (float(x_val), float(y_val))

        if not point_map:
            return None

        segments = []
        vertical_segments = {}
        for line in geometry.findall("./Lines/Line"):
            p1 = line.findtext("PointID1")
            p2 = line.findtext("PointID2")
            if p1 not in point_map or p2 not in point_map:
                continue

            x1, y1 = point_map[p1]
            x2, y2 = point_map[p2]

            if np.isclose(x1, x2):
                vertical_segments.setdefault(round(x1, 6), []).append(max(y1, y2))
                continue

            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            segments.append((x1, y1, x2, y2))

        if not segments:
            arr = np.array(list(point_map.values()), dtype=float)
            arr = arr[arr[:, 0].argsort()]
            return arr

        x_min = min(min(seg[0], seg[2]) for seg in segments)
        x_max = max(max(seg[0], seg[2]) for seg in segments)
        sample_x = np.linspace(x_min, x_max, n_samples)
        profile_y = np.full_like(sample_x, fill_value=-np.inf, dtype=float)

        for x1, y1, x2, y2 in segments:
            mask = (sample_x >= x1) & (sample_x <= x2)
            if not np.any(mask):
                continue
            t = (sample_x[mask] - x1) / (x2 - x1)
            y = y1 + t * (y2 - y1)
            profile_y[mask] = np.maximum(profile_y[mask], y)

        for idx, x_val in enumerate(sample_x):
            key = round(float(x_val), 6)
            if key in vertical_segments:
                profile_y[idx] = max(profile_y[idx], max(vertical_segments[key]))

        valid = np.isfinite(profile_y)
        if not np.any(valid):
            return None

        x_valid = sample_x[valid]
        y_valid = profile_y[valid]

        keep = [0]
        for i in range(1, len(x_valid) - 1):
            if not (np.isclose(y_valid[i], y_valid[i - 1], atol=1e-4) and np.isclose(y_valid[i], y_valid[i + 1], atol=1e-4)):
                keep.append(i)
        if len(x_valid) > 1:
            keep.append(len(x_valid) - 1)

        return np.column_stack([x_valid[keep], y_valid[keep]])
    except Exception:
        return None


def load_data(dataset_dir, mode):
    """Load dataset."""
    for name in [f'{mode}.csv', f'results_{mode}.csv', f'results_{mode}']:
        p = Path(dataset_dir) / name
        if p.exists():
            print(f"Loaded: {p.name}")
            return pd.read_csv(p)
    raise FileNotFoundError(f"No {mode} dataset")


def create_stats_table(df, mode, output_dir):
    """Statistics table with 4 categories."""
    
    df_ok = df[df['success'] == True]
    fos = df_ok['fos'].to_numpy()
    
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f'Statistical Summary - {mode.upper()} Mode',
                 fontsize=20, fontweight='bold', y=0.96, color='#000')
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # 4 SAFETY CATEGORIES
    n_failure = np.sum(fos < 1.0)
    n_critical = np.sum((fos >= 1.0) & (fos < 1.3))
    n_moderate = np.sum((fos >= 1.3) & (fos < 1.5))
    n_safe = np.sum(fos >= 1.5)
    
    data = [
        ['METRIC', 'VALUE', '', 'PERCENTILE', 'VALUE'],
        ['-'*20, '-'*15, '', '-'*20, '-'*15],
        ['Total Samples', f'{len(df)}', '', '5th', f'{np.percentile(fos, 5):.4f}'],
        ['Successful', f'{len(df_ok)}', '', '25th (Q1)', f'{np.percentile(fos, 25):.4f}'],
        ['Success Rate', f'{len(df_ok)/len(df)*100:.1f}%', '', '50th (Median)', f'{np.percentile(fos, 50):.4f}'],
        ['', '', '', '75th (Q3)', f'{np.percentile(fos, 75):.4f}'],
        ['', '', '', '95th', f'{np.percentile(fos, 95):.4f}'],
        ['', '', '', '', ''],
        ['FOS Minimum', f'{np.min(fos):.4f}', '', 'CATEGORY', 'COUNT'],
        ['FOS Maximum', f'{np.max(fos):.4f}', '', '-'*20, '-'*15],
        ['FOS Mean', f'{np.mean(fos):.4f}', '', 'Failure (<1.0)', f'{n_failure} ({n_failure/len(fos)*100:.1f}%)'],
        ['FOS Std Dev', f'{np.std(fos):.4f}', '', 'Critical (1.0-1.3)', f'{n_critical} ({n_critical/len(fos)*100:.1f}%)'],
        ['FOS Range', f'{np.max(fos)-np.min(fos):.4f}', '', 'Moderate (1.3-1.5)', f'{n_moderate} ({n_moderate/len(fos)*100:.1f}%)'],
        ['', '', '', 'Safe (>=1.5)', f'{n_safe} ({n_safe/len(fos)*100:.1f}%)'],
    ]
    
    table = ax.table(cellText=data, cellLoc='left',
                    bbox=[0.05, 0.20, 0.90, 0.70],
                    colWidths=[0.28, 0.22, 0.05, 0.28, 0.22])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3.0)
    
    for i, row in enumerate(data):
        for j in range(len(row)):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#34495E')
                cell.set_text_props(weight='bold', color='white', fontsize=13)
            elif '-' in str(row[j]):
                cell.set_facecolor('#ECF0F1')
            else:
                cell.set_facecolor('#FFFFFF' if i%2==0 else '#F8F9FA')
                if j in [0, 3]:
                    cell.set_text_props(weight='bold')
    
    out = output_dir / f'STATS_{mode.upper()}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    print(f"  Saved: {out.name}")
    
    return fig


def create_plots(df, mode, output_dir, control_points=None, geometry_profile=None,
                 title_suffix="", filename_suffix=""):
    """Analysis plots with 4-category coloring."""
    
    df_ok = df[df['success'] == True].copy()
    fos = df_ok['fos'].to_numpy()
    n = len(df_ok)
    
    has_surfaces = 'phreatic_surface_y' in df_ok.columns
    
    fig = plt.figure(figsize=(20, 12.5))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35, 
                          top=0.89, bottom=0.06, left=0.06, right=0.96)
    
    suffix_text = f" - {title_suffix}" if title_suffix else ""
    fig.suptitle(f'FOS Analysis - {mode.upper()} - {n} Samples{suffix_text}',
                 fontsize=24, fontweight='bold', y=0.965, color='#000')
    
    # PLOT 1: Phreatic Surfaces (Heatmapped)
    ax1 = fig.add_subplot(gs[0, :2])

    if geometry_profile is not None and len(geometry_profile) > 1:
        y_floor = np.min(geometry_profile[:, 1]) - 20.0
        ax1.fill_between(geometry_profile[:, 0], y_floor, geometry_profile[:, 1],
                         color='#c9b08b', alpha=0.18, zorder=0)
    
    if has_surfaces:
        fos_min = max(0.8, fos.min() - 0.05)
        fos_max = min(2.0, fos.max() + 0.05)
        
        cmap = cm.RdYlGn
        norm = Normalize(vmin=fos_min, vmax=fos_max)
        
        n_plotted = 0
        
        for _, row in df_ok.iterrows():
            try:
                surf = row['phreatic_surface_y']
                
                if pd.isna(surf) or not isinstance(surf, str):
                    continue
                
                y = json.loads(surf)
                if not y:
                    continue
                
                if control_points and 'interpolation' in control_points:
                    x_min = control_points['interpolation'].get('x_min', 0.0)
                    x_max = control_points['interpolation'].get('x_max', float(len(y) - 1))
                else:
                    x_min, x_max = 0.0, 1851.6
                x = np.linspace(x_min, x_max, len(y))
                color = cmap(norm(row['fos']))
                
                ax1.plot(x, y, alpha=0.6, linewidth=1.8, color=color)
                n_plotted += 1
                
            except:
                continue
        
        ax1.set_title(f'{n_plotted} Phreatic Surfaces (Heatmapped by FOS)', 
                      fontweight='bold', fontsize=14, pad=15)
        
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, pad=0.015, aspect=35)
        cbar.set_label('FOS Value', fontsize=13, fontweight='bold', rotation=270, labelpad=25)
        
    else:
        ax1.text(0.5, 0.5, 'No surface data', ha='center', va='center', fontsize=16)
    
    ax1.set_xlabel('Distance (ft)', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Elevation (ft)', fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.25)
    if geometry_profile is not None and len(geometry_profile) > 1:
        ax1.plot(geometry_profile[:, 0], geometry_profile[:, 1],
                 color='black', linewidth=3.2, alpha=0.98, label='Facility Geometry', zorder=10)
        ax1.legend(loc='lower left', fontsize=10, frameon=True)
    
    # PLOT 2: FOS Histogram (4 categories)
    ax2 = fig.add_subplot(gs[0, 2])
    
    bins = min(30, max(12, n//4))
    _, edges, patches = ax2.hist(fos, bins=bins, edgecolor='blue', linewidth=1.5, alpha=0.9)
    
    # Color by 4 CATEGORIES
    for patch, edge in zip(patches, edges[:-1]):
        idx = list(edges).index(edge)
        mid = (edge + edges[idx+1]) / 2
        
        if mid < 1.0:
            patch.set_facecolor('#E74C3C')      # Red
        elif mid < 1.3:
            patch.set_facecolor('#F39C12')      # Orange
        elif mid < 1.5:
            patch.set_facecolor('#3498DB')      # Blue
        else:
            patch.set_facecolor('#27AE60')      # Green
    
    ax2.axvline(np.mean(fos), color='#000', linestyle='--', linewidth=2.5, 
                label=f'Mean: {np.mean(fos):.3f}')
    ax2.axvline(1.0, color='#cc0000', linestyle='-', linewidth=2.5, label='Failure')
    ax2.axvline(1.3, color='#ff9900', linestyle=':', linewidth=2, label='Critical')
    ax2.axvline(1.5, color='#009900', linestyle='--', linewidth=2, label='Safe')
    
    ax2.set_xlabel('FOS', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Count', fontweight='bold', fontsize=13)
    ax2.set_title('FOS Distribution', fontweight='bold', fontsize=14, pad=15)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # PLOT 3: Control Points
    ax3 = fig.add_subplot(gs[1, 0])
    
    cp_cols = [c for c in df_ok.columns if c.startswith('control_y_')]
    cp_data = [df_ok[c].to_numpy() for c in cp_cols]
    
    ax3.boxplot(cp_data, tick_labels=[f'CP{i+1}' for i in range(len(cp_cols))],
                patch_artist=True, widths=0.6,
                boxprops=dict(facecolor='#3498DB', alpha=0.7, linewidth=1.5),
                medianprops=dict(color='darkred', linewidth=2.5))
    
    ax3.set_ylabel('Elevation (ft)', fontweight='bold', fontsize=13)
    ax3.set_xlabel('Control Point', fontweight='bold', fontsize=13)
    ax3.set_title('Control Point Ranges', fontweight='bold', fontsize=14, pad=15)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # PLOT 4: FOS vs CP1
    ax4 = fig.add_subplot(gs[1, 1])
    
    fos_min = max(0.8, fos.min() - 0.05)
    fos_max = min(2.0, fos.max() + 0.05)
    
    ax4.scatter(df_ok['control_y_1'], fos, c=fos, cmap='RdYlGn',
                s=80, alpha=0.7, edgecolors='black', linewidth=0.5,
                vmin=fos_min, vmax=fos_max)
    
    ax4.axhline(1.0, color='#cc0000', linestyle='-', linewidth=2.5, alpha=0.7)
    ax4.axhline(1.3, color='#ff9900', linestyle=':', linewidth=2, alpha=0.7)
    ax4.axhline(1.5, color='#009900', linestyle='--', linewidth=2, alpha=0.7)
    
    ax4.set_xlabel('Pond Elevation (ft)', fontweight='bold', fontsize=13)
    ax4.set_ylabel('FOS', fontweight='bold', fontsize=13)
    ax4.set_title('FOS vs Pond', fontweight='bold', fontsize=14, pad=15)
    ax4.grid(True, alpha=0.3)
    
    # PLOT 5: FOS vs Last CP
    ax5 = fig.add_subplot(gs[1, 2])
    
    last_cp = cp_cols[-1]
    
    ax5.scatter(df_ok[last_cp], fos, c=fos, cmap='RdYlGn',
                s=80, alpha=0.7, edgecolors='black', linewidth=0.5,
                vmin=fos_min, vmax=fos_max)
    
    ax5.axhline(1.0, color='#cc0000', linestyle='-', linewidth=2.5, alpha=0.7)
    ax5.axhline(1.3, color='#ff9900', linestyle=':', linewidth=2, alpha=0.7)
    ax5.axhline(1.5, color='#009900', linestyle='--', linewidth=2, alpha=0.7)
    
    ax5.set_xlabel('Downstream Elevation (ft)', fontweight='bold', fontsize=13)
    ax5.set_ylabel('FOS', fontweight='bold', fontsize=13)
    ax5.set_title('FOS vs Downstream', fontweight='bold', fontsize=14, pad=15)
    ax5.grid(True, alpha=0.3)
    
    suffix_file = f"_{filename_suffix}" if filename_suffix else ""
    out = output_dir / f'PLOTS_{mode.upper()}{suffix_file}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    print(f"  Saved: {out.name}")
    
    return fig


def print_summary(df, mode):
    """Print summary with 4 categories."""
    df_ok = df[df['success'] == True]
    fos = df_ok['fos'].to_numpy()
    
    print("\n" + "="*70)
    print(f"  FOS DATASET - {mode.upper()}")
    print("="*70)
    print(f"  Samples: {len(df)} ({len(df_ok)} successful, {len(df_ok)/len(df)*100:.1f}%)")
    print(f"  FOS Range: [{np.min(fos):.4f}, {np.max(fos):.4f}]")
    print(f"  Mean: {np.mean(fos):.4f} +/- {np.std(fos):.4f}")
    print(f"  Median: {np.median(fos):.4f}")
    
    # 4 CATEGORIES
    n_failure = np.sum(fos < 1.0)
    n_critical = np.sum((fos >= 1.0) & (fos < 1.3))
    n_moderate = np.sum((fos >= 1.3) & (fos < 1.5))
    n_safe = np.sum(fos >= 1.5)
    
    print(f"\n  Safety Categories:")
    print(f"    Failure (<1.0):    {n_failure:>4} ({n_failure/len(fos)*100:>5.1f}%)")
    print(f"    Critical (1.0-1.3): {n_critical:>4} ({n_critical/len(fos)*100:>5.1f}%)")
    print(f"    Moderate (1.3-1.5): {n_moderate:>4} ({n_moderate/len(fos)*100:>5.1f}%)")
    print(f"    Safe (>=1.5):      {n_safe:>4} ({n_safe/len(fos)*100:>5.1f}%)")
    print("="*70)


def main():
    """Main."""
    import argparse
    
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=str((project_root / 'data' / 'datasets' / 'default').resolve()))
    parser.add_argument('--mode', default='shallow', choices=['shallow', 'deep'])
    parser.add_argument('--config-path', type=str, default=None)
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("FOS DATASET VISUALIZATION")
    print("="*70)
    
    try:
        runtime = load_runtime_config(args.config_path)
        geometry_profile = load_geometry_profile(args.config_path)
        df = load_data(args.dataset, args.mode)
        print_summary(df, args.mode)
        
        outdir = Path(args.dataset)
        
        print("\nCreating statistics table...")
        f1 = create_stats_table(df, args.mode, outdir)
        
        print("Creating analysis plots...")
        f2 = create_plots(
            df,
            args.mode,
            outdir,
            control_points=runtime.get('control_points'),
            geometry_profile=geometry_profile,
        )

        df_low = df[df['fos'] < 1.25].copy()
        if len(df_low) > 0:
            print("Creating low-FOS analysis plots (<1.25)...")
            create_plots(
                df_low,
                args.mode,
                outdir,
                control_points=runtime.get('control_points'),
                geometry_profile=geometry_profile,
                title_suffix='FOS < 1.25',
                filename_suffix='LT125',
            )

        print("\n" + "="*70)
        print("COMPLETE!")
        print("="*70)
        print(f"  STATS_{args.mode.upper()}.png")
        print(f"  PLOTS_{args.mode.upper()}.png")
        if len(df_low) > 0:
            print(f"  PLOTS_{args.mode.upper()}_LT125.png")
        print("="*70 + "\n")
        
        plt.close('all')
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
