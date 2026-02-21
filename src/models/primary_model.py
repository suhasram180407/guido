"""
Step 3 – Primary Predictive Model
==================================
Two options:
  1. ElasticNet logistic regression (scikit-learn)  — default
  2. Shallow neural network (PyTorch)               — opt-in via USE_NEURAL_NET=True

The primary model is trained on the full training split after passing
a gate check: AUROC must be >= best baseline AUROC on the validation set.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import MLFLOW_TRACKING_URI, RANDOM_SEED
from src.models.metrics import compute_clf_metrics, log_metrics_to_mlflow

logger = logging.getLogger(__name__)

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ─── ElasticNet Logistic Regression ─────────────────────────────────────────

def build_elasticnet(C: float = 0.05, l1_ratio: float = 0.5) -> Pipeline:
    """
    Logistic Regression with ElasticNet penalty.
    Uses saga solver which supports both l1 and elasticnet.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            l1_ratio=l1_ratio,  # 0=ridge, 1=lasso; replaces deprecated penalty='elasticnet'
            C=C,
            max_iter=3000,
            tol=1e-4,
            random_state=RANDOM_SEED,
        )),
    ])


# ─── Shallow Neural Network (PyTorch) ────────────────────────────────────────

class ShallowNN:
    """
    Lightweight PyTorch MLP: Input → 256 → Dropout → 64 → Dropout → 1.
    Wrapped in a scikit-learn-compatible interface (fit / predict_proba).
    """

    def __init__(
        self,
        input_dim: int = 2000,
        hidden_dims: Tuple[int, ...] = (256, 64),
        dropout: float = 0.4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 10,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.model_ = None
        self.scaler_ = StandardScaler()

    def _build_net(self):
        import torch.nn as nn
        layers = []
        in_dim = self.input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShallowNN":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(RANDOM_SEED)
        X_sc = self.scaler_.fit_transform(X)
        X_t = torch.tensor(X_sc, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        self.input_dim = X.shape[1]
        self.model_ = self._build_net().to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model_(Xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                patience_counter = 0
                import copy
                self._best_state = copy.deepcopy(self.model_.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d (best_loss=%.4f)", epoch, best_loss)
                    break

        if hasattr(self, "_best_state"):
            self.model_.load_state_dict(self._best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.model_.eval()
        X_sc = self.scaler_.transform(X)
        X_t = torch.tensor(X_sc, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model_(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        # Return (n, 2) to match sklearn convention
        return np.stack([1 - probs, probs], axis=1)


# ─── Gate check ──────────────────────────────────────────────────────────────

def passes_gate(
    primary_auroc: float,
    baseline_results: Dict[str, Dict[str, float]],
) -> bool:
    """
    Returns True if primary model AUROC >= any baseline mean AUROC.
    """
    best_baseline = max(v["auroc_mean"] for v in baseline_results.values())
    if primary_auroc >= best_baseline:
        logger.info("Gate PASSED: primary AUROC %.4f >= best baseline %.4f", primary_auroc, best_baseline)
        return True
    logger.warning(
        "Gate FAILED: primary AUROC %.4f < best baseline %.4f", primary_auroc, best_baseline
    )
    return False


# ─── Train & evaluate primary model ─────────────────────────────────────────

def train_primary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    baseline_results: Dict[str, Dict[str, float]],
    use_neural_net: bool = False,
    save_path: Optional[Path] = None,
) -> Tuple[object, Dict[str, float], bool]:
    """
    Train primary model, evaluate on validation set, run gate check.

    Returns
    -------
    model        : trained model object (with predict_proba)
    val_metrics  : dict of validation metrics
    gate_passed  : bool
    """
    model_name = "ShallowNN" if use_neural_net else "ElasticNet"

    from contextlib import nullcontext
    ctx = mlflow.start_run(run_name=f"{model_name}_primary") if _MLFLOW_AVAILABLE else nullcontext()

    with ctx:
        if use_neural_net:
            model = ShallowNN(input_dim=X_train.shape[1])
            if _MLFLOW_AVAILABLE:
                mlflow.log_params({
                    "type": "ShallowNN", "hidden": str(model.hidden_dims),
                    "dropout": model.dropout, "lr": model.lr,
                    "weight_decay": model.weight_decay, "epochs": model.epochs,
                    "patience": model.patience,
                })
            model.fit(X_train.values, y_train.values)
            y_prob = model.predict_proba(X_val.values)[:, 1]
        else:
            model = build_elasticnet()
            if _MLFLOW_AVAILABLE:
                mlflow.log_params({"type": "ElasticNet", "C": 0.05, "l1_ratio": 0.5})
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]

        val_metrics = compute_clf_metrics(y_val, y_prob)
        log_metrics_to_mlflow(val_metrics, tags={"phase": "validation"})
        logger.info("%s validation metrics: %s", model_name, val_metrics)

        gate = passes_gate(val_metrics["auroc"], baseline_results)
        if _MLFLOW_AVAILABLE:
            mlflow.set_tag("gate_passed", str(gate))

        if save_path:
            import joblib
            save_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, save_path / f"{model_name}.joblib")
            if _MLFLOW_AVAILABLE:
                mlflow.log_artifact(str(save_path / f"{model_name}.joblib"))

    return model, val_metrics, gate
