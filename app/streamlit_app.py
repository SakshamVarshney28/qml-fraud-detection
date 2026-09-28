"""Static-artifact Streamlit demonstration for fraud detection results.

This version displays saved experiment outputs only. It does not fit models,
launch QSVM computation, or claim arbitrary new-transaction inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = PROJECT_ROOT / "results" / "json"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
HQNN_METRICS = PROJECT_ROOT / "results" / "hqnn-metrics"
HQNN_PREDICTIONS = PROJECT_ROOT / "results" / "hqnn-predictions"
HQNN_PLOTS = PROJECT_ROOT / "results" / "figures" / "hqnn-plots"
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
EXPERIMENTAL_FEATURES = ("Time", "Amount", "V17", "V14", "V12", "V10")

HQNN_CONFIG_LABELS = {
    "original": "Original (X0, X1)",
    "extended_x": "Extended-X (X0, X1, extended)",
    "extended_y": "Extended-Y (Y0, Y1)",
    "combined": "Combined (X0, X1, Y0, Y1)",
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_json(filename: str) -> dict:
    path = RESULTS_JSON / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required result file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_predictions(filename: str) -> pd.DataFrame:
    path = RESULTS_JSON / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required prediction file is missing: {path}")
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


@st.cache_data
def load_comparison() -> pd.DataFrame:
    path = RESULTS_TABLES / "midterm_model_comparison.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required comparison table is missing: {path}")
    return pd.read_csv(path)


@st.cache_data
def load_experimental_features() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset file is missing: {DATASET_PATH}")
    features = pd.read_csv(DATASET_PATH, usecols=[*EXPERIMENTAL_FEATURES, "Class"])
    features.index.name = "source_row_index"
    return features


@st.cache_data
def load_hqnn_metrics(config: str) -> dict:
    path = HQNN_METRICS / f"{config}_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"HQNN metrics file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_hqnn_history(config: str) -> dict:
    path = HQNN_METRICS / f"{config}_history.json"
    if not path.is_file():
        raise FileNotFoundError(f"HQNN history file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_hqnn_baselines() -> dict:
    path = HQNN_METRICS / "baselines_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"HQNN baselines file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_hqnn_predictions(config: str) -> pd.DataFrame:
    path = HQNN_PREDICTIONS / f"{config}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"HQNN predictions file missing: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def display_metric_cards(metrics: dict) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, f"{value:.4f}")


def display_figure(relative_path: str, caption: str) -> None:
    path = PROJECT_ROOT / relative_path
    if path.is_file():
        st.image(str(path), caption=caption)
    else:
        st.warning(f"Figure unavailable: {relative_path}")


def display_hqnn_figure(filename: str, caption: str) -> None:
    path = HQNN_PLOTS / filename
    if path.is_file():
        st.image(str(path), caption=caption)
    else:
        st.warning(f"HQNN figure unavailable: {filename}")


def status_label(value: int) -> str:
    return "Fraud" if value == 1 else "Legitimate"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_experimental_input_demo(
    predictions: pd.DataFrame,
    experimental_features: pd.DataFrame,
) -> None:
    """Browse saved held-out feature rows and compare saved predictions."""
    st.header("4. Experimental held-out input")
    st.caption(
        "Read-only demonstration using an existing held-out transaction. "
        "These controls do not run a new model prediction."
    )
    model_columns = {
        "Logistic Regression": ("logistic_regression_prediction", "logistic_regression_score"),
        "RBF-SVM": ("rbf_svm_prediction", "rbf_svm_score"),
        "XGBoost": ("xgboost_prediction", "xgboost_score"),
    }
    selected_model = st.selectbox(
        "Choose a classical model",
        list(model_columns),
        key="experimental_model",
    )
    selected_index = st.selectbox(
        "Choose an existing held-out transaction",
        predictions["source_row_index"].tolist(),
        format_func=lambda value: f"Transaction {value}",
        key="experimental_transaction",
    )
    row = predictions.loc[predictions["source_row_index"] == selected_index].iloc[0]
    feature_row = experimental_features.loc[int(selected_index)]

    with st.container(border=True):
        st.write("Saved feature values")
        feature_columns = st.columns(3)
        for position, feature_name in enumerate(EXPERIMENTAL_FEATURES):
            feature_columns[position % 3].number_input(
                feature_name,
                value=float(feature_row[feature_name]),
                disabled=True,
                key=f"experimental_{feature_name}",
            )
        st.caption(
            "Fields are read-only because no fitted model artifact for arbitrary new inputs is saved."
        )

    prediction_column, score_column = model_columns[selected_model]
    selected_result = pd.DataFrame([{
        "selected model": selected_model,
        "transaction index": int(selected_index),
        "actual status": f"Actual {status_label(row['true_class'])}",
        "prediction": f"Predicted {status_label(row[prediction_column])}",
        "score/probability": row[score_column],
    }])
    st.dataframe(selected_result, width="stretch", hide_index=True)

    if st.toggle(
        "Compare all classical models for this transaction",
        value=True,
        key="compare_experimental_models",
    ):
        comparison_rows = []
        for model_name, (prediction_name, score_name) in model_columns.items():
            comparison_rows.append({
                "model": model_name,
                "prediction": f"Predicted {status_label(row[prediction_name])}",
                "score/probability": row[score_name],
            })
        st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)


def render_classical_transaction_demo(predictions: pd.DataFrame) -> None:
    st.subheader("Transaction-level fraud detection demo")
    st.caption("Demonstration using existing held-out classical test predictions, not live inference for an arbitrary new transaction.")
    selected_index = st.selectbox(
        "Select a held-out transaction",
        predictions["source_row_index"].tolist(),
        format_func=lambda value: f"Transaction {value}",
    )
    row = predictions.loc[predictions["source_row_index"] == selected_index].iloc[0]
    st.write({
        "Transaction index": int(row["source_row_index"]),
        "Time": row.get("time", "Unavailable"),
        "Actual status": status_label(row["true_class"]),
    })
    model_columns = {
        "Logistic Regression": ("logistic_regression_prediction", "logistic_regression_score"),
        "RBF-SVM": ("rbf_svm_prediction", "rbf_svm_score"),
        "XGBoost": ("xgboost_prediction", "xgboost_score"),
    }
    display_rows = []
    for model, (prediction_column, score_column) in model_columns.items():
        display_rows.append({
            "model": model,
            "prediction": f"Predicted {status_label(row[prediction_column])}",
            "score/probability": row[score_column],
        })
    st.dataframe(pd.DataFrame(display_rows), width="stretch", hide_index=True)


def render_classical_section(classical_results: dict, predictions: pd.DataFrame) -> None:
    st.header("3. Classical fraud detection")
    st.caption("Logistic Regression, RBF-SVM, and XGBoost use the existing full baseline held-out predictions. No model is retrained here.")
    metric_rows = []
    for model, values in classical_results["models"].items():
        metric_rows.append({
            "model": model.replace("_", " ").title() if model != "rbf_svm" else "RBF-SVM",
            "PR-AUC": values["pr_auc"], "ROC-AUC": values["roc_auc"],
            "precision": values["precision"], "recall": values["recall"],
            "F1": values["f1"], "specificity": values["specificity"],
            "FPR": values["false_positive_rate"],
        })
    st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
    render_classical_transaction_demo(predictions)

    st.header("5. Fraud detection metrics")
    analysis = load_json("fraud_detection_analysis.json")
    st.dataframe(pd.DataFrame(analysis["fraud_detection_table"]), width="stretch", hide_index=True)
    confusion_columns = st.columns(3)
    for column, row in zip(confusion_columns, analysis["models"]):
        with column:
            st.write(row["model"])
            st.dataframe(
                pd.DataFrame(
                    [[row["TN"], row["FP"]], [row["FN"], row["TP"]]],
                    index=["Actual legitimate", "Actual fraud"],
                    columns=["Predicted legitimate", "Predicted fraud"],
                ),
                width="stretch",
            )


def render_qsvm_section(qsvm_results: dict, qsvm_predictions: pd.DataFrame) -> None:
    st.header("6. QSVM demonstration")
    st.warning("Preliminary reduced-subset QSVM. It uses saved 32/32 predictions only; it does not train QSVC or predict arbitrary new transactions.")
    config = qsvm_results["quantum_configuration"]
    st.write({
        "features": qsvm_results["feature_reduction"]["selected_features"],
        "feature count": config["feature_count"], "qubits": config["qubit_count"],
        "feature map": config["feature_map"], "reps": config["feature_map_reps"],
        "entanglement": config["feature_map_entanglement"], "sampler": config["sampler"],
        "shots": config["shots"], "fidelity": config["fidelity_implementation"],
        "kernel": config["kernel_type"], "classifier": "QSVC",
        "training samples": qsvm_results["training_subset"]["rows_used"],
        "test samples": qsvm_results["test_set"]["rows_used"],
    })
    metrics = qsvm_results["metrics"]
    display_metric_cards({"PR-AUC": metrics["pr_auc"], "ROC-AUC": metrics["roc_auc"], "Precision": metrics["precision"], "Recall": metrics["recall"], "F1": metrics["f1"]})
    st.write({"specificity": metrics["specificity"], "FPR": metrics["false_positive_rate"], "confusion matrix": metrics["confusion_matrix"]})
    selected_index = st.selectbox("Select an existing QSVM test prediction", qsvm_predictions["source_row_index"].tolist(), format_func=lambda value: f"Transaction {value}", key="qsvm_transaction")
    row = qsvm_predictions.loc[qsvm_predictions["source_row_index"] == selected_index].iloc[0]
    st.write({"transaction index": int(row["source_row_index"]), "actual Class": int(row["true_class"]), "QSVM prediction": int(row["qsvc_prediction"]), "QSVM score": row["qsvc_score"]})


def render_hqnn_section() -> None:
    """Render the full HQNN results section using only hqnn-metrics, hqnn-predictions, and hqnn-plots."""
    st.header("7. Hybrid Quantum Neural Network (HQNN)")
    st.info(
        "Photonic HQNN with 4 quantum layers and 2 modes, evaluated across four measurement "
        "configurations. Results loaded from saved experiment artifacts only — no model is "
        "retrained or re-evaluated here."
    )

    configs = list(HQNN_CONFIG_LABELS.keys())
    selected_config = st.segmented_control(
        "Measurement configuration",
        options=configs,
        format_func=lambda c: HQNN_CONFIG_LABELS[c],
        default="combined",
        key="hqnn_config_selector",
    )
    if selected_config is None:
        selected_config = "combined"

    metrics = load_hqnn_metrics(selected_config)
    history = load_hqnn_history(selected_config)

    with st.container(border=True):
        info_cols = st.columns(4)
        info_cols[0].metric("Quantum layers", metrics["model_info"]["num_quantum_layers"])
        info_cols[1].metric("Modes", metrics["model_info"]["num_modes"])
        info_cols[2].metric("Measurements", metrics["model_info"]["num_measurements"])
        info_cols[3].metric("Best threshold", metrics["best_threshold"])

    st.caption(f"Measurement operators: {', '.join(metrics['model_info']['measurement_ops'])}")

    st.subheader("Test set performance")
    tm = metrics["test_metrics"]
    display_metric_cards({
        "PR-AUC": tm["pr_auc"],
        "ROC-AUC": tm["roc_auc"],
        "F1": tm["f1"],
        "Precision": tm["precision"],
        "Recall": tm["recall"],
        "Accuracy": tm["accuracy"],
    })

    st.subheader("Confusion matrix")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
        index=["Actual legitimate", "Actual fraud"],
        columns=["Predicted legitimate", "Predicted fraud"],
    )
    cm_cols = st.columns([1, 2])
    with cm_cols[0]:
        st.dataframe(cm_df, width="stretch")
    with cm_cols[1]:
        st.write({
            "TP (true positives)": cm["TP"],
            "TN (true negatives)": cm["TN"],
            "FP (false positives)": cm["FP"],
            "FN (false negatives)": cm["FN"],
            "training time (s)": round(metrics["training_info"]["training_time_seconds"], 4),
        })

    st.subheader("Training history")
    history_df = pd.DataFrame({
        "epoch": list(range(1, len(history["loss"]) + 1)),
        "train loss": history["loss"],
        "val loss": history["val_loss"],
        "train accuracy": history["accuracy"],
        "val accuracy": history["val_accuracy"],
    }).set_index("epoch")
    tab_loss, tab_acc = st.tabs(["Loss", "Accuracy"])
    with tab_loss:
        st.line_chart(history_df[["train loss", "val loss"]])
    with tab_acc:
        st.line_chart(history_df[["train accuracy", "val accuracy"]])

    st.subheader("All HQNN configurations — metric comparison")
    comparison_rows = []
    for cfg in configs:
        try:
            m = load_hqnn_metrics(cfg)["test_metrics"]
            mi = load_hqnn_metrics(cfg)["model_info"]
            comparison_rows.append({
                "config": HQNN_CONFIG_LABELS[cfg],
                "measurements": mi["num_measurements"],
                "measurement ops": ", ".join(mi["measurement_ops"]),
                "PR-AUC": round(m["pr_auc"], 6),
                "ROC-AUC": round(m["roc_auc"], 6),
                "F1": round(m["f1"], 6),
                "precision": round(m["precision"], 6),
                "recall": round(m["recall"], 6),
                "accuracy": round(m["accuracy"], 6),
            })
        except FileNotFoundError:
            pass
    if comparison_rows:
        st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

    st.subheader("HQNN-notebook baselines (LR & RF)")
    st.caption("These baseline results were computed inside the HQNN experiment notebook on the same dataset split.")
    try:
        baselines = load_hqnn_baselines()
        baseline_rows = []
        for model_key, vals in baselines.items():
            baseline_rows.append({
                "model": model_key.replace("_", " ").title(),
                "PR-AUC": round(vals["pr_auc"], 6),
                "ROC-AUC": round(vals["roc_auc"], 6),
                "F1": round(vals["f1"], 6),
                "accuracy": round(vals["accuracy"], 6),
            })
        st.dataframe(pd.DataFrame(baseline_rows), width="stretch", hide_index=True)
    except FileNotFoundError as e:
        st.warning(str(e))

    st.subheader("Transaction-level HQNN prediction demo")
    st.caption(
        f"Browsing saved held-out predictions from the '{HQNN_CONFIG_LABELS[selected_config]}' "
        "configuration. No model is run here."
    )
    try:
        hqnn_preds = load_hqnn_predictions(selected_config)
        idx_col = next(
            (c for c in ["index", "source_row_index", "transaction_index"] if c in hqnn_preds.columns),
            hqnn_preds.columns[0],
        )
        selected_tx = st.selectbox(
            "Select a held-out HQNN transaction",
            hqnn_preds[idx_col].tolist(),
            format_func=lambda v: f"Transaction {v}",
            key="hqnn_transaction_selector",
        )
        tx_row = hqnn_preds.loc[hqnn_preds[idx_col] == selected_tx].iloc[0]
        tx_dict = {col: tx_row[col] for col in hqnn_preds.columns}
        st.write(tx_dict)
    except FileNotFoundError as e:
        st.warning(str(e))

    st.subheader("HQNN experiment figures")
    st.caption("All figures are saved artifacts from the HQNN experiment.")
    plot_grid = [
        ("training_loss.png", "Training & validation loss"),
        ("training_accuracy.png", "Training & validation accuracy"),
        ("roc_curve.png", "ROC curve"),
        ("precision_recall_curve.png", "Precision-recall curve"),
        ("confusion_matrix.png", "Confusion matrix (counts)"),
        ("confusion_matrix_normalized.png", "Confusion matrix (normalised)"),
        ("threshold_analysis.png", "Threshold analysis"),
        ("class_distribution.png", "Class distribution"),
    ]
    col1, col2 = st.columns(2)
    for i, (fname, cap) in enumerate(plot_grid):
        with (col1 if i % 2 == 0 else col2):
            display_hqnn_figure(fname, cap)


def render_model_comparison_section(comparison: pd.DataFrame) -> None:
    """Render the unified model comparison including HQNN results."""
    st.header("8. Model comparison")

    hqnn_rows = []
    for cfg in HQNN_CONFIG_LABELS:
        try:
            m = load_hqnn_metrics(cfg)["test_metrics"]
            hqnn_rows.append({
                "Model": f"HQNN ({HQNN_CONFIG_LABELS[cfg]})",
                "PR-AUC": round(m["pr_auc"], 6),
                "ROC-AUC": round(m["roc_auc"], 6),
                "F1": round(m["f1"], 6),
                "Precision": round(m["precision"], 6),
                "Recall": round(m["recall"], 6),
                "Accuracy": round(m["accuracy"], 6),
            })
        except FileNotFoundError:
            pass

    st.subheader("Midterm classical + QSVM comparison")
    st.dataframe(comparison, width="stretch", hide_index=True)
    st.info(
        "The classical models use the full baseline train/test evaluation. "
        "The preliminary QSVM uses reduced 32/32 subsets, so its metric values are not directly "
        "equivalent to the classical full-data results."
    )

    if hqnn_rows:
        st.subheader("HQNN configurations vs. HQNN-notebook baselines")
        st.caption(
            "HQNN results are from the hqnn-metrics saved artifacts. "
            "The HQNN-notebook baselines (LR & RF) below were computed on the same dataset split as HQNN."
        )
        st.markdown("**HQNN configurations**")
        st.dataframe(pd.DataFrame(hqnn_rows), width="stretch", hide_index=True)

        try:
            baselines = load_hqnn_baselines()
            bl_rows = []
            for model_key, vals in baselines.items():
                bl_rows.append({
                    "Model": f"Baseline — {model_key.replace('_', ' ').title()}",
                    "PR-AUC": round(vals["pr_auc"], 6),
                    "ROC-AUC": round(vals["roc_auc"], 6),
                    "F1": round(vals["f1"], 6),
                    "Accuracy": round(vals["accuracy"], 6),
                })
            st.markdown("**HQNN-notebook baselines (same split)**")
            st.dataframe(pd.DataFrame(bl_rows), width="stretch", hide_index=True)
        except FileNotFoundError:
            pass

        st.subheader("Best HQNN config at a glance")
        best = max(hqnn_rows, key=lambda r: r["PR-AUC"])
        best_cols = st.columns(6)
        for col, (k, v) in zip(best_cols, [
            ("Model", best["Model"]),
            ("PR-AUC", f"{best['PR-AUC']:.4f}"),
            ("ROC-AUC", f"{best['ROC-AUC']:.4f}"),
            ("F1", f"{best['F1']:.4f}"),
            ("Precision", f"{best['Precision']:.4f}"),
            ("Recall", f"{best['Recall']:.4f}"),
        ]):
            col.metric(k, v)


def main() -> None:
    st.set_page_config(page_title="Fraud Detection Demonstration", layout="wide")
    st.title(
        "Noise-Aware Hybrid Quantum-Classical Financial Fraud Detection "
        "Under Class Imbalance and Temporal Distribution Shift"
    )
    try:
        dataset = load_json("dataset_summary.json")
        classical_results = load_json("classical_results.json")
        qsvm_results = load_json("qsvm_results.json")
        classical_predictions = load_predictions("classical_test_predictions.json")
        qsvm_predictions = load_predictions("qsvm_predictions.json")
        comparison = load_comparison()
        experimental_features = load_experimental_features()
    except (FileNotFoundError, json.JSONDecodeError) as error:
        st.error(f"Saved artifact unavailable: {error}")
        st.stop()

    st.header("1. Project overview")
    st.write(
        "This is a binary fraud-detection task: Class 1 is fraud and Class 0 is legitimate. "
        "The dataset is severely imbalanced, so PR-AUC / Average Precision is emphasized."
    )

    st.header("2. Dataset summary")
    st.write({
        "total transactions": dataset["row_count"],
        "fraud transactions": dataset["class_distribution"]["1"],
        "legitimate transactions": dataset["class_distribution"]["0"],
        "fraud percentage": dataset["fraud_percentage"],
    })

    render_experimental_input_demo(classical_predictions, experimental_features)
    render_classical_section(classical_results, classical_predictions)
    render_qsvm_section(qsvm_results, qsvm_predictions)
    render_hqnn_section()
    render_model_comparison_section(comparison)

    st.header("9. Existing figures (classical & QSVM)")
    for relative_path, caption in [
        ("results/figures/eda/01_class_distribution.png", "EDA: class distribution"),
        ("results/figures/classical/01_pr_curves.png", "Classical PR curves"),
        ("results/figures/classical/02_roc_curves.png", "Classical ROC curves"),
        ("results/figures/classical/03_confusion_matrices.png", "Classical confusion matrices"),
        ("results/figures/fraud_detection/fraud_detected_missed.png", "Frauds actually present, detected, and missed"),
        ("results/figures/fraud_detection/legitimate_transactions_flagged.png", "Legitimate transactions falsely flagged"),
        ("results/figures/qsvm/01_pr_curve.png", "QSVM PR curve"),
        ("results/figures/qsvm/02_roc_curve.png", "QSVM ROC curve"),
        ("results/figures/qsvm/04_quantum_kernel_matrix.png", "QSVM training kernel matrix"),
        ("results/figures/midterm_model_comparison.png", "Factual model comparison"),
    ]:
        display_figure(relative_path, caption)

    st.header("10. Research status")
    st.markdown(
        "**COMPLETED:** dataset validation, EDA, preprocessing, classical models, quantum feature "
        "reduction, QSVM sanity check, preliminary QSVM, fraud-detection analysis, midterm "
        "comparison, HQNN (4 measurement configurations), and this artifact-based demonstration.\n\n"
        "**PRELIMINARY:** reduced-subset QSVM.\n\n"
        "**FUTURE:** chronological/temporal distribution shift; simulated quantum noise; 4/6/8-qubit "
        "scaling; computational resource analysis; explainability; optional VQC; inference-ready saved "
        "model artifacts; true new-transaction interactive prediction mode."
    )


if __name__ == "__main__":
    main()
