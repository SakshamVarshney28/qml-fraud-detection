"""Fixed, reproducible classical baseline model definitions.

The functions here create models only. Fitting and evaluation are intentionally
kept separate so every model can receive the same predefined train/test split.
No hyperparameter search is performed in the midterm baseline.
"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


RANDOM_SEED = 42
RBF_SVM_MAX_ITERATIONS = 10_000


def build_logistic_regression(random_state: int = RANDOM_SEED) -> LogisticRegression:
    """Create a class-weighted Logistic Regression baseline with fixed settings."""

    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_state,
        solver="lbfgs",
    )


def build_rbf_svm(random_state: int = RANDOM_SEED) -> SVC:
    """Create an RBF-SVM baseline with a recorded, fixed iteration cap.

    Full-kernel SVM fitting is computationally demanding at this dataset size.
    The cap is a fixed resource safeguard, not a tuned hyperparameter. The
    experiment records ``fit_status_`` so a non-converged run is never hidden.
    """

    return SVC(
        C=1.0,
        cache_size=1_024,
        class_weight="balanced",
        gamma="scale",
        kernel="rbf",
        max_iter=RBF_SVM_MAX_ITERATIONS,
        random_state=random_state,
    )


def build_xgboost(
    scale_pos_weight: float, random_state: int = RANDOM_SEED
) -> XGBClassifier:
    """Create a class-weighted histogram XGBoost baseline with fixed settings."""

    return XGBClassifier(
        colsample_bytree=0.8,
        eval_metric="logloss",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=300,
        n_jobs=-1,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        tree_method="hist",
    )


def calculate_scale_pos_weight(target) -> float:
    """Calculate the XGBoost class weight from training labels only."""

    positive_count = int((target == 1).sum())
    negative_count = int((target == 0).sum())
    if positive_count == 0:
        raise ValueError("Training target contains no positive fraud examples.")
    return negative_count / positive_count


def model_configuration(model_name: str, model: Any) -> Dict[str, Any]:
    """Return serialisable fixed settings for a constructed baseline model."""

    parameters = model.get_params()
    if model_name == "logistic_regression":
        selected = ("C", "class_weight", "max_iter", "random_state", "solver")
    elif model_name == "rbf_svm":
        selected = (
            "C",
            "cache_size",
            "class_weight",
            "gamma",
            "kernel",
            "max_iter",
            "random_state",
        )
    elif model_name == "xgboost":
        selected = (
            "colsample_bytree",
            "eval_metric",
            "learning_rate",
            "max_depth",
            "n_estimators",
            "n_jobs",
            "random_state",
            "scale_pos_weight",
            "subsample",
            "tree_method",
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return {name: parameters[name] for name in selected}


def build_classical_models(training_target) -> Dict[str, Any]:
    """Build all fixed baselines using training labels only where needed."""

    scale_pos_weight = calculate_scale_pos_weight(training_target)
    return {
        "logistic_regression": build_logistic_regression(),
        "rbf_svm": build_rbf_svm(),
        "xgboost": build_xgboost(scale_pos_weight=scale_pos_weight),
    }
