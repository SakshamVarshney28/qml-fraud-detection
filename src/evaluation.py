"""Evaluation, artefact saving, and experiment runner for classical baselines.

PR-AUC (average precision) is the primary metric. This module never chooses a
threshold from test labels: each fitted model's native prediction rule supplies
the class prediction while ranking scores supply PR-AUC and ROC-AUC.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from classical_models import build_classical_models, model_configuration
from data import load_dataset
from preprocessing import PreparedDataset, preprocessing_configuration, prepare_baseline_data


MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "rbf_svm": "RBF-SVM",
    "xgboost": "XGBoost",
}
MODEL_COLORS = {
    "logistic_regression": "#4C78A8",
    "rbf_svm": "#F58518",
    "xgboost": "#54A24B",
}


def _number(value: Any) -> int | float:
    """Convert numpy scalar values to JSON-safe built-in numbers."""

    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def get_ranking_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return model ranking scores without consulting true test labels."""

    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1])
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features))
    raise TypeError("Model must expose predict_proba or decision_function for ranking metrics.")


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    """Calculate required classification metrics from fixed model predictions."""

    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    score_values = np.asarray(y_score)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        true_values, predicted_values, labels=[0, 1]
    ).ravel()
    specificity = true_negative / (true_negative + false_positive)
    false_positive_rate = false_positive / (false_positive + true_negative)

    return {
        "pr_auc": _number(average_precision_score(true_values, score_values)),
        "roc_auc": _number(roc_auc_score(true_values, score_values)),
        "precision": _number(precision_score(true_values, predicted_values, zero_division=0)),
        "recall": _number(recall_score(true_values, predicted_values, zero_division=0)),
        "f1": _number(f1_score(true_values, predicted_values, zero_division=0)),
        "specificity": _number(specificity),
        "false_positive_rate": _number(false_positive_rate),
        "confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
    }


def fit_and_evaluate_model(
    model_name: str,
    model: Any,
    prepared: PreparedDataset,
) -> tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Fit one model on the common training set and evaluate the common test set."""

    fit_warnings: list[str] = []
    training_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(prepared.X_train_transformed, prepared.split.y_train)
    training_time_seconds = time.perf_counter() - training_start
    fit_warnings = [
        str(item.message)
        for item in caught_warnings
        if issubclass(item.category, ConvergenceWarning)
    ]

    inference_start = time.perf_counter()
    predictions = np.asarray(model.predict(prepared.X_test_transformed))
    scores = get_ranking_scores(model, prepared.X_test_transformed)
    inference_time_seconds = time.perf_counter() - inference_start

    evaluation = evaluate_predictions(prepared.split.y_test, predictions, scores)
    evaluation["training_time_seconds"] = _number(training_time_seconds)
    evaluation["inference_time_seconds"] = _number(inference_time_seconds)
    evaluation["fit_warnings"] = fit_warnings
    fit_status = getattr(model, "fit_status_", None)
    evaluation["fit_status"] = int(fit_status) if fit_status is not None else None
    evaluation["model_configuration"] = model_configuration(model_name, model)
    return evaluation, predictions, scores


def prediction_records(
    prepared: PreparedDataset,
    predictions_by_model: Dict[str, np.ndarray],
    scores_by_model: Dict[str, np.ndarray],
) -> list[Dict[str, Any]]:
    """Create one row per shared test example with each model's actual outputs."""

    rows: list[Dict[str, Any]] = []
    for position, source_index in enumerate(prepared.X_test_transformed.index):
        record: Dict[str, Any] = {
            "source_row_index": int(source_index),
            "time": _number(prepared.split.time_test.loc[source_index]),
            "true_class": int(prepared.split.y_test.loc[source_index]),
        }
        for model_name in predictions_by_model:
            record[f"{model_name}_prediction"] = int(predictions_by_model[model_name][position])
            record[f"{model_name}_score"] = _number(scores_by_model[model_name][position])
        rows.append(record)
    return rows


def save_json(content: Any, output_path: str | Path) -> None:
    """Save serialisable experimental artefacts as UTF-8 JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(content, file_handle, indent=2)
        file_handle.write("\n")


def plot_pr_curves(
    y_true: pd.Series,
    scores_by_model: Dict[str, np.ndarray],
    metrics_by_model: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save precision-recall curves with measured average precision labels."""

    figure, axis = plt.subplots(figsize=(9, 6))
    for model_name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        axis.plot(
            recall,
            precision,
            label=f"{MODEL_DISPLAY_NAMES[model_name]} (AP={metrics_by_model[model_name]['pr_auc']:.4f})",
            color=MODEL_COLORS[model_name],
        )
    baseline = float(y_true.mean())
    axis.axhline(baseline, color="gray", linestyle="--", label=f"Prevalence ({baseline:.4f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curves (Primary Metric: PR-AUC)")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_roc_curves(
    y_true: pd.Series,
    scores_by_model: Dict[str, np.ndarray],
    metrics_by_model: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save ROC curves with actual ROC-AUC labels."""

    figure, axis = plt.subplots(figsize=(9, 6))
    for model_name, scores in scores_by_model.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{MODEL_DISPLAY_NAMES[model_name]} (AUC={metrics_by_model[model_name]['roc_auc']:.4f})",
            color=MODEL_COLORS[model_name],
        )
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC Curves")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrices(
    metrics_by_model: Dict[str, Dict[str, Any]], output_path: Path
) -> None:
    """Save one clearly labelled confusion-matrix panel per baseline model."""

    figure, axes = plt.subplots(1, len(metrics_by_model), figsize=(15, 4.5))
    for axis, (model_name, metrics) in zip(axes, metrics_by_model.items()):
        matrix = metrics["confusion_matrix"]
        array = np.array(
            [
                [matrix["true_negative"], matrix["false_positive"]],
                [matrix["false_negative"], matrix["true_positive"]],
            ]
        )
        sns.heatmap(
            array,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted non-fraud", "Predicted fraud"],
            yticklabels=["Actual non-fraud", "Actual fraud"],
            ax=axis,
        )
        axis.set_title(MODEL_DISPLAY_NAMES[model_name])
        axis.set_xlabel("Prediction")
        axis.set_ylabel("Actual class")
    figure.suptitle("Confusion Matrices at Each Model's Native Prediction Rule")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_pr_auc_comparison(metrics_by_model: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """Save a direct comparison of the primary PR-AUC metric."""

    model_names = list(metrics_by_model)
    values = [metrics_by_model[name]["pr_auc"] for name in model_names]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        [MODEL_DISPLAY_NAMES[name] for name in model_names],
        values,
        color=[MODEL_COLORS[name] for name in model_names],
    )
    axis.set_xlabel("Model")
    axis.set_ylabel("PR-AUC (average precision)")
    axis.set_title("Classical Baseline Comparison: Primary PR-AUC")
    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.4f}",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_classical_figures(
    prepared: PreparedDataset,
    scores_by_model: Dict[str, np.ndarray],
    metrics_by_model: Dict[str, Dict[str, Any]],
    output_directory: str | Path,
) -> list[str]:
    """Generate all required classical-baseline comparison figures."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    figures = {
        "01_pr_curves.png": plot_pr_curves,
        "02_roc_curves.png": plot_roc_curves,
    }
    for filename, function in figures.items():
        function(prepared.split.y_test, scores_by_model, metrics_by_model, directory / filename)
    plot_confusion_matrices(metrics_by_model, directory / "03_confusion_matrices.png")
    plot_pr_auc_comparison(metrics_by_model, directory / "04_pr_auc_comparison.png")
    return [
        "01_pr_curves.png",
        "02_roc_curves.png",
        "03_confusion_matrices.png",
        "04_pr_auc_comparison.png",
    ]


def run_classical_baselines(
    dataset_path: str | Path,
    results_output_path: str | Path,
    predictions_output_path: str | Path,
    figures_output_directory: str | Path,
) -> Dict[str, Any]:
    """Run all fixed classical baselines on one common prepared split."""

    dataset = load_dataset(dataset_path)
    prepared = prepare_baseline_data(dataset)
    models = build_classical_models(prepared.split.y_train)

    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    predictions_by_model: Dict[str, np.ndarray] = {}
    scores_by_model: Dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        metrics, predictions, scores = fit_and_evaluate_model(model_name, model, prepared)
        metrics_by_model[model_name] = metrics
        predictions_by_model[model_name] = predictions
        scores_by_model[model_name] = scores

    figure_names = save_classical_figures(
        prepared, scores_by_model, metrics_by_model, figures_output_directory
    )
    predictions = prediction_records(prepared, predictions_by_model, scores_by_model)
    save_json(predictions, predictions_output_path)

    result = {
        "primary_metric": "pr_auc (average precision)",
        "threshold_policy": "Native model predictions; no threshold selected using test labels.",
        "preprocessing": preprocessing_configuration(prepared, dataset_path),
        "models": metrics_by_model,
        "prediction_file": str(predictions_output_path),
        "figure_files": figure_names,
    }
    save_json(result, results_output_path)
    return result


def main() -> None:
    """Execute the fixed classical-baseline experiment without test-set tuning."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to creditcard.csv")
    parser.add_argument("--results", type=Path, required=True, help="JSON metrics output")
    parser.add_argument("--predictions", type=Path, required=True, help="JSON test predictions output")
    parser.add_argument("--figures-dir", type=Path, required=True, help="Directory for output figures")
    arguments = parser.parse_args()

    result = run_classical_baselines(
        dataset_path=arguments.dataset,
        results_output_path=arguments.results,
        predictions_output_path=arguments.predictions,
        figures_output_directory=arguments.figures_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
