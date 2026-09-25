"""Small, noiseless QSVC sanity check for the preliminary quantum pipeline.

This module intentionally does not run a full quantum-kernel experiment. A
standard kernel SVM requires a kernel matrix whose size grows quadratically with
the number of training examples, so the sanity check uses a fixed, explicit
stratified subset of the existing training split. The complete baseline test
split remains the evaluation set.

Architecture:
    baseline features -> training-only reduction -> ZZ feature map
    -> fidelity quantum kernel -> QSVC

The implementation uses current Qiskit 2.x and Qiskit Machine Learning APIs.
It does not use QuantumInstance, QuantumKernel, qiskit.opflow, or Aer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from data import load_dataset
from evaluation import evaluate_predictions
from preprocessing import PreparedDataset, prepare_baseline_data
from quantum_features import reduce_to_quantum_features


RANDOM_SEED = 42
FEATURE_COUNT = 4
TRAINING_SUBSET_SIZE = 400
TEST_FRAUD_COUNT = 50
TEST_LEGITIMATE_COUNT = 5000
SAMPLER_SHOTS = 1024
FEATURE_MAP_REPS = 2
FEATURE_MAP_ENTANGLEMENT = "full"


def _number(value: Any) -> int | float:
    """Convert a numeric value into a JSON-safe built-in number."""

    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def select_training_subset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    subset_size: int = TRAINING_SUBSET_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select a fixed-size class-balanced subset from training data only."""

    if subset_size < 2:
        raise ValueError("subset_size must be at least 2.")
    if subset_size > len(X_train):
        raise ValueError("subset_size cannot exceed the training split size.")
    class_values = sorted(pd.Series(y_train).unique().tolist())
    if len(class_values) < 2:
        raise ValueError("Training data must contain at least two classes.")
    if subset_size % len(class_values) != 0:
        raise ValueError("subset_size must divide evenly across target classes.")

    samples_per_class = subset_size // len(class_values)
    rng = np.random.default_rng(random_state)
    selected_indices: list[Any] = []
    for class_value in class_values:
        class_indices = pd.Index(y_train.index[np.asarray(y_train) == class_value])
        if samples_per_class > len(class_indices):
            raise ValueError("A target class has too few training rows for this subset.")
        selected_indices.extend(
            rng.choice(class_indices.to_numpy(), size=samples_per_class, replace=False).tolist()
        )
    subset_indices = pd.Index(selected_indices).sort_values()
    return X_train.loc[subset_indices].copy(), y_train.loc[subset_indices].copy()


def select_test_subset(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fraud_count: int = TEST_FRAUD_COUNT,
    legitimate_count: int = TEST_LEGITIMATE_COUNT,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select a deterministic imbalanced subset from baseline test data only."""

    rng = np.random.default_rng(random_state)
    
    fraud_indices = pd.Index(y_test.index[np.asarray(y_test) == 1])
    legit_indices = pd.Index(y_test.index[np.asarray(y_test) == 0])
    
    if fraud_count > len(fraud_indices):
        raise ValueError(f"Requested {fraud_count} fraud samples, but only {len(fraud_indices)} available.")
    if legitimate_count > len(legit_indices):
        raise ValueError(f"Requested {legitimate_count} legit samples, but only {len(legit_indices)} available.")

    selected_fraud = rng.choice(fraud_indices.to_numpy(), size=fraud_count, replace=False).tolist()
    selected_legit = rng.choice(legit_indices.to_numpy(), size=legitimate_count, replace=False).tolist()
    
    subset_indices = pd.Index(selected_fraud + selected_legit).sort_values()
    return X_test.loc[subset_indices].copy(), y_test.loc[subset_indices].copy()


def build_quantum_kernel(
    feature_count: int = FEATURE_COUNT,
    reps: int = FEATURE_MAP_REPS,
    entanglement: str = FEATURE_MAP_ENTANGLEMENT,
    shots: int = SAMPLER_SHOTS,
    random_state: int = RANDOM_SEED,
) -> tuple[Any, Any, Any, Any]:
    """Construct the current noiseless feature-map, fidelity, and kernel stack."""

    feature_map = zz_feature_map(
        feature_dimension=feature_count,
        reps=reps,
        entanglement=entanglement,
    )
    sampler = StatevectorSampler(default_shots=shots, seed=random_state)
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    return feature_map, sampler, fidelity, kernel


def save_kernel_heatmap(kernel_matrix: np.ndarray, output_path: str | Path) -> None:
    """Save a simple heatmap of the small training kernel matrix."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(kernel_matrix, cmap="viridis", vmin=0, vmax=1)
    axis.set_title("Small Quantum Training Kernel Matrix")
    axis.set_xlabel("Training subset sample")
    axis.set_ylabel("Training subset sample")
    figure.colorbar(image, ax=axis, label="Kernel fidelity")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_json(content: Any, output_path: str | Path) -> None:
    """Save a JSON experiment artifact."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(content, file_handle, indent=2)
        file_handle.write("\n")


def save_prediction_records(
    prepared: PreparedDataset,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save predictions for rows in the sanity-check test subset."""

    records = [
        {
            "source_row_index": int(source_index),
            "time": _number(prepared.split.time_test.loc[source_index]),
            "true_class": int(prepared.split.y_test.loc[source_index]),
            "qsvc_prediction": int(predictions[position]),
            "qsvc_score": _number(scores[position]),
        }
        for position, source_index in enumerate(test_features.index)
    ]
    save_json(records, output_path)


def save_evaluation_figures(
    y_true: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray,
    feature_map: Any,
    training_kernel_matrix: np.ndarray,
    output_directory: str | Path,
) -> dict[str, str]:
    """Save preliminary PR, ROC, confusion-matrix, and circuit figures."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, label="QSVC")
    axis.axhline(float(y_true.mean()), color="gray", linestyle="--", label="Prevalence")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Preliminary QSVC Precision-Recall Curve")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    pr_path = directory / "01_pr_curve.png"
    figure.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(false_positive_rate, true_positive_rate, label="QSVC")
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Preliminary QSVC ROC Curve")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    roc_path = directory / "02_roc_curve.png"
    figure.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    axis.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    axis.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    axis.set_title("Preliminary QSVC Confusion Matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    confusion_path = directory / "03_confusion_matrix.png"
    figure.savefig(confusion_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(training_kernel_matrix, cmap="viridis", vmin=0, vmax=1)
    axis.set_title("Preliminary QSVC Training Kernel Matrix")
    axis.set_xlabel("Training subset sample")
    axis.set_ylabel("Training subset sample")
    figure.colorbar(image, ax=axis, label="Kernel fidelity")
    figure.tight_layout()
    heatmap_path = directory / "04_quantum_kernel_matrix.png"
    figure.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    circuit_path = directory / "05_quantum_feature_map.png"
    try:
        circuit_figure = feature_map.draw(output="mpl", fold=-1)
        circuit_figure.savefig(circuit_path, dpi=200, bbox_inches="tight")
        plt.close(circuit_figure)
    except Exception:
        circuit_path = directory / "05_quantum_feature_map.txt"
        circuit_path.write_text(str(feature_map.draw(output="text", fold=-1)) + "\n", encoding="utf-8")

    figure_paths = {
        "pr_curve": str(pr_path),
        "roc_curve": str(roc_path),
        "confusion_matrix": str(confusion_path),
        "quantum_kernel_matrix": str(heatmap_path),
    }
    if circuit_path is not None:
        figure_paths["quantum_feature_map"] = str(circuit_path)
    return figure_paths


def run_qsvc_sanity_check(
    dataset_path: str | Path,
    results_output_path: str | Path,
    predictions_output_path: str | Path,
    heatmap_output_path: str | Path,
    subset_size: int = TRAINING_SUBSET_SIZE,
    test_fraud_count: int = TEST_FRAUD_COUNT,
    test_legitimate_count: int = TEST_LEGITIMATE_COUNT,
    figure_output_directory: str | Path | None = None,
    experiment_name: str = "small_noiseless_qsvc_sanity_check",
) -> dict[str, Any]:
    """Run the small QSVC check on a subset of the common baseline test partition."""

    experiment_start = time.perf_counter()
    dataset = load_dataset(dataset_path)
    prepared = prepare_baseline_data(dataset)
    selector, reduced_train, reduced_test = reduce_to_quantum_features(
        prepared.X_train_transformed,
        prepared.split.y_train,  # type: ignore[arg-type]
        prepared.X_test_transformed,
        feature_count=FEATURE_COUNT,
    )
    subset_train, subset_target = select_training_subset(
        reduced_train,
        prepared.split.y_train,
        subset_size=subset_size,
    )
    subset_test, subset_test_target = select_test_subset(
        reduced_test,
        prepared.split.y_test,
        fraud_count=test_fraud_count,
        legitimate_count=test_legitimate_count,
    )

    kernel_construction_start = time.perf_counter()
    feature_map, sampler, fidelity, quantum_kernel = build_quantum_kernel()
    kernel_construction_time = time.perf_counter() - kernel_construction_start

    kernel_matrix_start = time.perf_counter()
    training_kernel_matrix = quantum_kernel.evaluate(subset_train.to_numpy())
    kernel_matrix_time = time.perf_counter() - kernel_matrix_start
    save_kernel_heatmap(training_kernel_matrix, heatmap_output_path)

    qsvc = QSVC(
        quantum_kernel=quantum_kernel,
        C=10.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    training_start = time.perf_counter()
    qsvc.fit(subset_train, subset_target)
    training_time = time.perf_counter() - training_start

    inference_start = time.perf_counter()
    scores = np.asarray(qsvc.decision_function(subset_test))
    predictions = (scores > 0).astype(int)
    inference_time = time.perf_counter() - inference_start

    metrics = evaluate_predictions(subset_test_target, predictions, scores)
    metrics["kernel_matrix_time_seconds"] = _number(kernel_matrix_time)
    metrics["training_time_seconds"] = _number(training_time)
    metrics["inference_time_seconds"] = _number(inference_time)
    figure_files = {}
    if figure_output_directory is not None:
        figure_files = save_evaluation_figures(
            subset_test_target,
            predictions,
            scores,
            feature_map,
            training_kernel_matrix,
            figure_output_directory,
        )
    save_prediction_records(
        prepared,
        subset_test,
        subset_test_target,
        predictions,
        scores,
        predictions_output_path,
    )

    result = {
        "experiment": experiment_name,
        "status": "completed",
        "architecture": [
            "4 classical features",
            "4 qubits",
            "zz_feature_map",
            "FidelityQuantumKernel",
            "QSVC",
        ],
        "reproducibility": {
            "random_state": RANDOM_SEED,
            "sampler_shots": SAMPLER_SHOTS,
            "training_subset_selection": "stratified subset from baseline X_train only",
            "test_subset_selection": "class-balanced subset from baseline X_test only",
            "baseline_split": "80/20 stratified random split, random_state=42",
        },
        "feature_reduction": {
            "method": "training-only ANOVA F-score selector",
            "feature_count": FEATURE_COUNT,
            "selected_features": list(selector.selected_feature_names),
            "training_rows_before_subset": len(reduced_train),
            "test_rows": len(reduced_test),
        },
        "quantum_configuration": {
            "feature_count": feature_map.num_parameters,
            "qubit_count": feature_map.num_qubits,
            "feature_map": "zz_feature_map",
            "feature_map_reps": FEATURE_MAP_REPS,
            "feature_map_entanglement": FEATURE_MAP_ENTANGLEMENT,
            "feature_map_parameters": [str(parameter) for parameter in feature_map.parameters],
            "kernel_type": type(quantum_kernel).__name__,
            "fidelity_implementation": type(fidelity).__name__,
            "sampler": type(sampler).__name__,
            "shots": SAMPLER_SHOTS,
        },
        "training_subset": {
            "selection": "stratified subset selected from existing baseline training split",
            "rows_used": len(subset_train),
            "class_distribution": {
                str(class_value): count
                for class_value, count in subset_target.value_counts().sort_index().items()
            },
        },
        "test_set": {
            "rows_in_baseline_split": len(reduced_test),
            "rows_used": len(subset_test),
            "source": "existing baseline test split",
            "label": "Reduced-subset preliminary QSVM experiment; QSVM sanity-check test subset — not the final evaluation protocol.",
            "random_state": RANDOM_SEED,
            "class_distribution": {
                str(class_value): count
                for class_value, count in subset_test_target.value_counts().sort_index().items()
            },
        },
        "kernel_workload_estimate": {
            "training_kernel_pairs": len(subset_train) ** 2,
            "inference_train_test_pairs_upper_bound": len(subset_train) * len(subset_test),
            "selection_reason": "32 training and 32 test rows balance repeatability with a small quantum-kernel workload.",
        },
        "timing_seconds": {
            "kernel_construction": _number(kernel_construction_time),
            "training_kernel_matrix": _number(kernel_matrix_time),
            "qsvc_training": _number(training_time),
            "inference": _number(inference_time),
        },
        "kernel_matrix": {
            "shape": list(training_kernel_matrix.shape),
            "heatmap_file": str(heatmap_output_path),
        },
        "metrics": metrics,
        "prediction_file": str(predictions_output_path),
        "figure_files": figure_files,
        "total_runtime_seconds": _number(time.perf_counter() - experiment_start),
        "limitations": [
            "This is a reduced-subset QSVM experiment using 400 training and 5050 test rows.",
            "The reduced-subset QSVM is not equivalent to full-data classical evaluation.",
            "It is not a full-dataset QSVM experiment and does not support a quantum-advantage claim.",
            "The run is noiseless and uses StatevectorSampler; no Aer or hardware noise is included.",
        ],
    }
    save_json(result, results_output_path)
    return result


def main() -> None:
    """Run the small QSVC sanity check from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--heatmap", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=TRAINING_SUBSET_SIZE)
    parser.add_argument("--test-fraud-count", type=int, default=TEST_FRAUD_COUNT)
    parser.add_argument("--test-legit-count", type=int, default=TEST_LEGITIMATE_COUNT)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", default="small_noiseless_qsvc_sanity_check")
    arguments = parser.parse_args()

    result = run_qsvc_sanity_check(
        dataset_path=arguments.dataset,
        results_output_path=arguments.results,
        predictions_output_path=arguments.predictions,
        heatmap_output_path=arguments.heatmap,
        subset_size=arguments.subset_size,
        test_fraud_count=arguments.test_fraud_count,
        test_legitimate_count=arguments.test_legit_count,
        figure_output_directory=arguments.figures_dir,
        experiment_name=arguments.experiment_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()