"""
Dataset Cleaning Script - With File Dialog
Removes samples that violate upstream > downstream flow rule
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime
from tkinter import Tk, filedialog

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def select_dataset_file():
    """Open file dialog to select dataset CSV."""
    root = Tk()
    root.withdraw()  # Hide main window
    root.attributes('-topmost', True)  # Bring to front
    
    print("\n📂 Select dataset CSV file...")
    
    file_path = filedialog.askopenfilename(
        title='Select FOS Dataset CSV File',
        filetypes=[
            ('CSV files', '*.csv'),
            ('All files', '*.*')
        ],
        initialdir=str((PROJECT_ROOT / 'data' / 'datasets').resolve())
    )
    
    root.destroy()
    
    if not file_path:
        print("❌ No file selected")
        return None
    
    print(f"✓ Selected: {Path(file_path).name}")
    return file_path


def check_flow_direction(row, strict=True):
    """
    Check if sample follows upstream > downstream rule.
    
    Args:
        row: DataFrame row with control_y_1, control_y_2, etc.
        strict: If True, enforce monotonic decrease
                If False, just check first > last
    
    Returns:
        (is_valid, violation_message)
    """
    # Extract control points
    cp_cols = [c for c in row.index if c.startswith('control_y_')]
    cp_values = [row[col] for col in cp_cols]
    
    if len(cp_values) < 2:
        return True, "OK"
    
    if strict:
        # Check monotonic non-increasing (each point <= previous)
        for i in range(1, len(cp_values)):
            if cp_values[i] > cp_values[i-1] + 0.01:  # Allow tiny tolerance
                return False, f"CP{i+1} ({cp_values[i]:.1f}) > CP{i} ({cp_values[i-1]:.1f})"
        return True, "OK"
    
    else:
        # Just check upstream > downstream
        if cp_values[-1] > cp_values[0]:
            return False, f"Downstream ({cp_values[-1]:.1f}) > Upstream ({cp_values[0]:.1f})"
        return True, "OK"


def clean_dataset(csv_path, strict=True, create_backup=True):
    """
    Clean dataset by removing invalid samples.
    
    Args:
        csv_path: Path to dataset CSV
        strict: Enforce strict monotonic decrease
        create_backup: Create backup before modifying
    
    Returns:
        (n_removed, n_kept)
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    # Load dataset
    print(f"\n📊 Loading dataset...")
    df = pd.read_csv(csv_path)
    n_original = len(df)
    print(f"  Original samples: {n_original}")
    
    # Create backup
    if create_backup:
        backup_path = csv_path.with_suffix('.csv.backup')
        shutil.copy(csv_path, backup_path)
        print(f"  ✓ Backup created: {backup_path.name}")
    
    # Check each sample
    print(f"\n🔍 Checking flow direction (strict={strict})...")
    
    valid_mask = []
    violations = []
    
    for idx, row in df.iterrows():
        is_valid, msg = check_flow_direction(row, strict=strict)
        valid_mask.append(is_valid)
        
        if not is_valid:
            violations.append({
                'sample_id': row.get('sample_id', idx),
                'reason': msg,
                'control_points': [row.get(f'control_y_{i+1}', np.nan) for i in range(5)]
            })
    
    # Filter dataset
    df_clean = df[valid_mask].copy()
    n_removed = n_original - len(df_clean)
    n_kept = len(df_clean)
    
    # Report
    print(f"\n" + "="*70)
    print(f"📋 CLEANING RESULTS")
    print(f"="*70)
    print(f"  Original samples: {n_original}")
    print(f"  Removed: {n_removed} ({n_removed/n_original*100:.1f}%)")
    print(f"  Kept: {n_kept} ({n_kept/n_original*100:.1f}%)")
    
    if n_removed > 0:
        print(f"\n🔍 Sample violations:")
        for v in violations[:10]:  # Show first 10
            cp_str = [f'{cp:.1f}' for cp in v['control_points'] if not np.isnan(cp)]
            print(f"  Sample {v['sample_id']}: {v['reason']}")
            print(f"    Control points: {cp_str}")
        
        if len(violations) > 10:
            print(f"  ... and {len(violations)-10} more")
    
    # Save cleaned dataset
    if n_removed > 0:
        response = input(f"\n💾 Save cleaned dataset? (y/n): ").lower().strip()
        
        if response == 'y':
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_path = csv_path.parent / f"{csv_path.stem}_clean.csv"
            
            df_clean.to_csv(clean_path, index=False)
            
            print(f"  ✓ Cleaned dataset saved: {clean_path.name}")
            print(f"  ✓ Original preserved as: {csv_path.name}")
            
            # Option to replace original
            replace = input(f"\n⚠️  Replace original file? (y/n): ").lower().strip()
            if replace == 'y':
                df_clean.to_csv(csv_path, index=False)
                print(f"  ✓ Original file updated")
                print(f"  ✓ Backup available: {backup_path.name}")
        else:
            print("  Cancelled - no changes saved")
    else:
        print("\n✅ Dataset is already clean! No violations found.")
    
    print("\n" + "="*70)
    
    return n_removed, n_kept


def main():
    """Main entry point."""
    print("="*70)
    print("🧹 FOS DATASET CLEANING TOOL")
    print("="*70)
    
    # Select file
    csv_path = select_dataset_file()
    
    if csv_path is None:
        print("\nExiting...")
        return
    
    # Ask for strict mode
    print("\n" + "="*70)
    print("CLEANING MODE:")
    print("  [1] STRICT - Enforce monotonic CP1 ≥ CP2 ≥ CP3 ≥ CP4 ≥ CP5")
    print("  [2] RELAXED - Only check upstream > downstream overall")
    print("="*70)
    
    mode_choice = input("Select mode (1 or 2): ").strip()
    strict = (mode_choice == '1')
    
    print(f"\n✓ Using {'STRICT' if strict else 'RELAXED'} mode")
    
    # Clean dataset
    try:
        n_removed, n_kept = clean_dataset(csv_path, strict=strict, create_backup=True)
        
        print("\n✅ Cleaning complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
