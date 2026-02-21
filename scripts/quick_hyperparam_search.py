#!/usr/bin/env python3
"""Quick hyperparameter search on a reduced gene set.

Selects top `n_genes` by variance from processed data (or from raw processed
matrix), runs a small GridSearchCV for ElasticNet, saves best model and
evaluation to `results/<project>/` and `models_artifacts/<project>/`.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(project: str, n_genes: int = 500):
    import sys
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    from src.api.pipeline import _load_splits
    from config.settings import RESULTS_DIR, MODELS_DIR
    from src.models.metrics import compute_clf_metrics

    # load splits cached earlier
    splits = _load_splits(project)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # select top n_genes by variance from combined train+val+test feature set
    X_all = pd.concat([X_train, X_val, X_test], axis=0)
    variances = X_all.var(axis=0)
    top_genes = variances.nlargest(n_genes).index.tolist()

    X_train_sub = X_train.loc[:, top_genes]
    X_val_sub = X_val.loc[:, top_genes]
    X_test_sub = X_test.loc[:, top_genes]

    logger.info("Running GridSearch on %d genes: %s...", n_genes, top_genes[:5])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000),
        ),
    ])
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0],
        "clf__l1_ratio": [0.2, 0.5, 0.8],
    }

    gs = GridSearchCV(pipe, param_grid, cv=3, scoring="roc_auc", n_jobs=1)
    gs.fit(X_train_sub, y_train)

    logger.info("Best params: %s", gs.best_params_)

    # Evaluate on test
    best = gs.best_estimator_
    if hasattr(best, "predict_proba"):
        y_prob = best.predict_proba(X_test_sub)[:, 1]
    else:
        # use decision function or predict and map
        try:
            scores = best.decision_function(X_test_sub)
            y_prob = 1 / (1 + np.exp(-scores))
        except Exception:
            y_prob = best.predict(X_test_sub)

    metrics = compute_clf_metrics(y_test.values, y_prob)
    logger.info("Test metrics: %s", metrics)

    # Save model and metrics
    mdl_dir = Path(MODELS_DIR) / project
    mdl_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, mdl_dir / "primary_model_tuned.joblib")

    out_dir = Path(RESULTS_DIR) / project
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tuned_params.json").write_text(json.dumps(gs.best_params_, indent=2))
    (out_dir / "eval_test_tuned.json").write_text(json.dumps(metrics, indent=2))

    print("Best params:", json.dumps(gs.best_params_, indent=2))
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--n-genes", type=int, default=500)
    args = parser.parse_args()
    main(args.project, args.n_genes)
