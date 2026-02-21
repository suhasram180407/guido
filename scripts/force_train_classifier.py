#!/usr/bin/env python3
"""Train a LogisticRegression classifier using cached splits and tuned params.

This forces a classifier model that supports `predict_proba`, saves it as
`models_artifacts/<project>/primary_model.joblib`, and writes evaluation JSON.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main(project: str):
    repo = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(repo))

    from src.api.pipeline import _load_splits
    from config.settings import MODELS_DIR, RESULTS_DIR
    from src.models.metrics import compute_clf_metrics

    splits = _load_splits(project)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    params_path = Path(RESULTS_DIR) / project / "tuned_params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text())
        C = float(params.get("clf__C", 1.0))
        l1_ratio = float(params.get("clf__l1_ratio", 0.5))
    else:
        C = 1.0
        l1_ratio = 0.5

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio, max_iter=5000),
        ),
    ])

    # Fit on train+val to use more data
    X_fit = np.vstack([X_train.values, X_val.values])
    y_fit = np.concatenate([y_train.values, y_val.values])
    pipe.fit(X_fit, y_fit)

    # Save model
    mdl_dir = Path(MODELS_DIR) / project
    mdl_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, mdl_dir / "primary_model.joblib")

    # Evaluate on test
    y_prob = pipe.predict_proba(X_test)[:, 1]
    metrics = compute_clf_metrics(y_test.values, y_prob)

    out_dir = Path(RESULTS_DIR) / project
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_test.json").write_text(json.dumps(metrics, indent=2))

    print("Saved primary_model.joblib and eval_test.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    main(args.project)
