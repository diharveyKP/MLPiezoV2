"""
Test XML template parsing using repo-local template paths.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATE_SHALLOW = PROJECT_ROOT / "models" / "templates" / "shallow.xml"
TEMPLATE_DEEP = PROJECT_ROOT / "models" / "templates" / "deep.xml"

print("="*70)
print("🔍 TESTING XML TEMPLATE PARSING")
print("="*70)

def test_template(template_path, template_name):
    """Test parsing a single template."""
    print(f"\n[TEST] {template_name} template:")
    print(f"  Path: {template_path}")
    
    # Check file exists
    template_path = Path(template_path)
    if not template_path.exists():
        print(f"  ❌ File not found!")
        print(f"     Expected template at: {template_path}")
        return False
    
    file_size = Path(template_path).stat().st_size
    print(f"  ✓ File exists ({file_size:,} bytes)")
    
    # Parse XML
    try:
        tree = ET.parse(template_path)
        root = tree.getroot()
        print("  ✓ XML parsed successfully")
    except ET.ParseError as e:
        print(f"  ❌ XML parsing error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    
    # Find DataPoints in StabilityItems
    try:
        datapoints = root.find('.//StabilityItems/StabilityItem/Entry/DataPoints')
        
        if datapoints is None:
            print("  ❌ DataPoints element not found!")
            return False
        
        dp_len = datapoints.get('Len')
        print(f"  ✓ Found DataPoints (Len={dp_len})")
        
        # List each DataPoint
        for dp in datapoints.findall('DataPoint'):
            num = dp.get('Number')
            x = dp.get('X')
            y = dp.get('Y')
            print(f"    DataPoint {num}: X={x}, Y={y}")
        
        # Find PiezometricSurface references
        piezo = root.find('.//PiezometricSurfaces/PiezometricSurface/DataPoints')
        if piezo:
            refs = [dp.text for dp in piezo.findall('DataPoint')]
            print(f"  ✓ PiezometricSurface references: {refs}")
        
        # Find slip surface settings
        slip_surface = root.find('.//StabilityItems/StabilityItem/Entry/SlipSurface')
        if slip_surface:
            min_depth = slip_surface.find('MinimumDepth')
            if min_depth is not None:
                print(f"  ✓ Slip surface MinimumDepth: {min_depth.text} ft")
            
            entry_exit = slip_surface.find('EntryExit')
            if entry_exit:
                left_pt = entry_exit.find('LeftSideLeftPt')
                right_pt = entry_exit.find('RightSideLeftPt')
                if left_pt is not None:
                    print(f"  ✓ Entry zone starts at X={left_pt.get('X')}")
                if right_pt is not None:
                    print(f"  ✓ Exit zone starts at X={right_pt.get('X')}")
        
        print(f"  ✅ {template_name} template OK!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error analyzing XML: {e}")
        import traceback
        traceback.print_exc()
        return False


# Run tests
print("\n" + "="*70)

shallow_ok = test_template(TEMPLATE_SHALLOW, "SHALLOW")
deep_ok = test_template(TEMPLATE_DEEP, "DEEP")

print("\n" + "="*70)
print("📊 TEST SUMMARY")
print("="*70)
print(f"Shallow template: {'✅ PASS' if shallow_ok else '❌ FAIL'}")
print(f"Deep template: {'✅ PASS' if deep_ok else '❌ FAIL'}")

if shallow_ok and deep_ok:
    print("\n✅ ALL TESTS PASSED!")
    print("\n🚀 READY FOR STEP 8: First Test Run")
    print("\nNext command:")
    print(f"  cd \"{PROJECT_ROOT}\"")
    print("  py -3.13 pipelines\\pipeline_generate_dataset.py --n_samples 1")
else:
    print("\n⚠️ FIX ERRORS ABOVE BEFORE PROCEEDING")
    if not shallow_ok:
        print("  → Check shallow template exists at specified path")
    if not deep_ok:
        print("  → Check deep template exists at specified path")

print("="*70)
