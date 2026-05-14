"""
Test script to verify project setup.
"""

import sys
from pathlib import Path

OK = "[OK]"
FAIL = "[FAIL]"

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def test_imports():
    """Test that all required packages are available."""
    print("Testing imports...")

    try:
        import numpy as np  # noqa: F401
        print(f"{OK} numpy")
    except ImportError:
        print(f"{FAIL} numpy")
        return False

    try:
        import pandas as pd  # noqa: F401
        print(f"{OK} pandas")
    except ImportError:
        print(f"{FAIL} pandas")
        return False

    try:
        import pydantic  # noqa: F401
        print(f"{OK} pydantic")
    except ImportError:
        print(f"{FAIL} pydantic")
        return False

    try:
        import yaml  # noqa: F401
        print(f"{OK} pyyaml")
    except ImportError:
        print(f"{FAIL} pyyaml")
        return False

    print(f"\n{OK} All imports successful")
    return True


def test_folders():
    """Test that all folders exist."""
    print("\nTesting folder structure...")

    required_folders = [
        "configs",
        "src",
        "pipelines",
        "scripts",
        "data/datasets",
        "logs",
        "models",
        "results/figures",
        "results/tables",
    ]

    all_exist = True
    for folder in required_folders:
        folder_path = project_root / folder
        if folder_path.exists():
            print(f"{OK} {folder}")
        else:
            print(f"{FAIL} {folder} - missing")
            all_exist = False

    if all_exist:
        print(f"\n{OK} All folders exist")
    else:
        print(f"\n{FAIL} Some folders are missing")

    return all_exist


if __name__ == "__main__":
    print("=" * 60)
    print("MLPiezoV2 - Setup Verification")
    print("=" * 60)
    print(f"\nProject root: {project_root}")

    imports_ok = test_imports()
    folders_ok = test_folders()

    print("\n" + "=" * 60)
    if imports_ok and folders_ok:
        print("SUCCESS! Your project setup is complete.")
        print("\nYou are ready to bootstrap a model workspace.")
    else:
        print("ISSUES DETECTED - Please fix the errors above")
    print("=" * 60)
