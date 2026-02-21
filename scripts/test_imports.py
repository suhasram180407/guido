#!/usr/bin/env python3
import sys
from pathlib import Path

print("Step 1: Adding repo to path...")
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
print(f"Path: {repo}")

print("Step 2: Importing settings...")
try:
    from config.settings import MODELS_DIR, RESULTS_DIR
    print(f"✓ Settings: MODELS_DIR={MODELS_DIR}, RESULTS_DIR={RESULTS_DIR}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("Step 3: Importing pipeline functions...")
try:
    from src.api.pipeline import _load_splits
    print("✓ Pipeline functions loaded")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("Step 4: Loading data splits...")
try:
    splits = _load_splits("TCGA-LUAD")
    print(f"✓ Splits loaded: train {splits['train'][0].shape}, val {splits['val'][0].shape}, test {splits['test'][0].shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\nAll imports successful!")
