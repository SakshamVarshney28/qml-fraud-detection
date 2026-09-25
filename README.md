# Noise-Aware Hybrid Quantum-Classical Financial Fraud Detection

University research project: **"Noise-Aware Hybrid Quantum-Classical Financial Fraud Detection Under Class Imbalance and Temporal Distribution Shift."**

The dataset location is `data/raw/creditcard.csv` (ULB Credit Card Fraud Detection). Raw and derived data are deliberately ignored by Git to prevent accidental dataset commits. The dataset is not redistributed by this repository.

## Current scope: midterm baseline preparation

Dataset validation and EDA are complete. The repository now contains a leakage-free preprocessing pipeline for the midterm baseline split. No model has been trained and no performance result is present.

The planned **midterm** work is:

1. Dataset validation and EDA.
2. Leakage-free preprocessing.
3. Logistic Regression, RBF-SVM, and XGBoost baselines.
4. Small quantum feature representation and preliminary QSVM/QSVC.
5. Reproducible comparisons, figures, and tables.

Temporal shift, simulated noise, qubit scaling, resource analysis, explainability, optional VQC, and Streamlit are explicitly out of scope until requested.

## Environment

Current Qiskit 2.x and Qiskit Machine Learning 0.9.x require Python 3.10 or later. Create a Python 3.10+ environment before installing project dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` records the intended project environment; it has not been installed as part of initialization.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/raw/` | Local, unchanged source data (including `creditcard.csv`). |
| `data/interim/` | Local intermediate data created during reproducible preparation. |
| `data/processed/` | Local model-ready datasets created only from training-fitted steps. |
| `src/fraud_detection/` | Reusable, tested Python package code. |
| `notebooks/` | Thin, narrative notebooks for EDA, experiments, and figures. |
| `configs/` | Version-controlled experiment configurations and random seeds. |
| `results/figures/` | Generated figures. |
| `results/tables/` | Generated tables and actual metrics. |
| `results/json/` | Generated machine-readable validation and experiment metadata. |
| `results/predictions/` | Generated prediction files, where practical. |
| `tests/` | Lightweight automated checks for repository code and layout. |

## Reproducibility and evaluation

All experiments must use fixed, recorded seeds and save their configuration with their actual outputs. PR-AUC is the primary metric. ROC-AUC, precision, recall, F1, specificity, FPR, and secondary accuracy will also be reported. The held-out test set must never influence preprocessing, feature selection, hyperparameter search, or threshold selection.

See [AGENTS.md](AGENTS.md) for the full project guardrails.

## Midterm preprocessing protocol

`src/preprocessing.py` defines the target as `Class` and retains all 30 non-target columns as baseline features: `Time`, `V1`–`V28`, and `Amount`. The conventional midterm evaluation uses an 80/20 stratified random split with `random_state=42`; it is explicitly separate from the future chronological experiment.

`StandardScaler` standardizes every numerical baseline feature, including `Amount`, but is fitted only on `X_train`. The test partition is transformed only after fitting and neither training nor test labels are passed to the preprocessor. The original unscaled `Time` values are retained with each split for the later chronological, non-shuffled protocol. The actual split configuration is saved to `results/json/preprocessing_config.json`.

## Next step

Review the saved preprocessing configuration and leakage tests before implementing any baseline model. Do not train models until explicitly requested.
