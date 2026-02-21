#!/usr/bin/env python3
import sys
try:
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    print('✓ imblearn installed')
except ImportError as e:
    print('✗ imblearn missing:', e)
    sys.exit(1)

try:
    from xgboost import XGBClassifier
    print('✓ xgboost installed')
except ImportError:
    print('✗ xgboost not installed (optional, will skip)')

print('All required libraries available!')
