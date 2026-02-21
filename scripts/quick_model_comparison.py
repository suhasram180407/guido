#!/usr/bin/env python3
"""Quick synchronous model comparison and performance improvement."""
import json
from pathlib import Path
import sys
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from src.api.pipeline import _load_splits
from config.settings import MODELS_DIR, RESULTS_DIR
from src.models.metrics import compute_clf_metrics

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBALANCED = True
except ImportError:
    HAS_IMBALANCED = False
    print("Warning: imbalanced-learn not available, skipping SMOTE")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Info: xgboost not available")

# Load data
splits = _load_splits("TCGA-LUAD")
X_train, y_train = splits["train"]
X_val, y_val = splits["val"]
X_test, y_test = splits["test"]

X_fit = pd.concat([X_train, X_val], axis=0)
y_fit = pd.concat([y_train, y_val], axis=0)

n_genes = 500
variances = X_fit.var(axis=0)
top_genes = variances.nlargest(n_genes).index.tolist()

X_fit_sub = X_fit.loc[:, top_genes]
X_test_sub = X_test.loc[:, top_genes]

print(f"Data: {X_fit_sub.shape[0]} train+val, {X_test_sub.shape[0]} test")
print(f"Class balance: {y_fit.value_counts().to_dict()}")

# Test models
results = {}

# Model 1: LogisticRegression with StandardScaler
print("\n1. Testing LogisticRegression...")
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="elasticnet", solver="saga", C=1.0, l1_ratio=0.5, max_iter=5000))
])
lr_scores = cross_val_score(lr_pipe, X_fit_sub, y_fit, cv=3, scoring="roc_auc")
results["LogisticRegression"] = {"cv_auroc_mean": float(lr_scores.mean()), "cv_auroc_std": float(lr_scores.std())}
print(f"   CV AUROC: {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})")

# Model 2: RandomForest
print("2. Testing RandomForest...")
rf_pipe = Pipeline([
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
])
rf_scores = cross_val_score(rf_pipe, X_fit_sub, y_fit, cv=3, scoring="roc_auc")
results["RandomForest"] = {"cv_auroc_mean": float(rf_scores.mean()), "cv_auroc_std": float(rf_scores.std())}
print(f"   CV AUROC: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")

# Model 3: XGBoost (if available)
if HAS_XGBOOST:
    print("3. Testing XGBoost...")
    xgb_pipe = Pipeline([
        ("clf", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss", verbosity=0))
    ])
    xgb_scores = cross_val_score(xgb_pipe, X_fit_sub, y_fit, cv=3, scoring="roc_auc")
    results["XGBoost"] = {"cv_auroc_mean": float(xgb_scores.mean()), "cv_auroc_std": float(xgb_scores.std())}
    print(f"   CV AUROC: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")

# Select best and evaluate on test
best_model_name = max(results.keys(), key=lambda x: results[x]["cv_auroc_mean"])
best_auroc_cv = results[best_model_name]["cv_auroc_mean"]

print(f"\n=== Best Model (CV): {best_model_name} (AUROC: {best_auroc_cv:.4f}) ===")

# Train best model on full data
if best_model_name == "LogisticRegression":
    best_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga", C=1.0, l1_ratio=0.5, max_iter=5000))
    ])
elif best_model_name == "RandomForest":
    best_model = Pipeline([
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
else:  # XGBoost
    best_model = Pipeline([
        ("clf", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss", verbosity=0))
    ])

best_model.fit(X_fit_sub, y_fit)
y_prob = best_model.predict_proba(X_test_sub)[:, 1]
test_metrics = compute_clf_metrics(y_test.values, y_prob)

print(f"\nTest Metrics:")
print(f"  AUROC: {test_metrics.get('auroc', 'N/A')}")
print(f"  Accuracy: {test_metrics.get('accuracy', 'N/A')}")
print(f"  F1: {test_metrics.get('f1', 'N/A')}")

# Save model and results
mdl_dir = Path(MODELS_DIR) / "TCGA-LUAD"
mdl_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, mdl_dir / "primary_model_improved.joblib")

out_dir = Path(RESULTS_DIR) / "TCGA-LUAD"
out_dir.mkdir(parents=True, exist_ok=True)

results["best_model"] = best_model_name
results["cv_auroc"] = float(best_auroc_cv)
results["test_metrics"] = test_metrics

(out_dir / "model_comparison_results.json").write_text(json.dumps(results, indent=2))

print(f"\n✓ Saved: {mdl_dir / 'primary_model_improved.joblib'}")
print(f"✓ Saved: {out_dir / 'model_comparison_results.json'}")
