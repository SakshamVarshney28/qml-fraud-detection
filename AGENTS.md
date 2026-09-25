# Research Project Guardrails

## Project

This repository supports the university research project **"Noise-Aware Hybrid Quantum-Classical Financial Fraud Detection Under Class Imbalance and Temporal Distribution Shift"** using the ULB Credit Card Fraud Detection dataset.

## Non-negotiable research rules

- Never invent results, metrics, citations, dataset statistics, or experimental outcomes.
- Never hardcode experimental results.
- Never use the test set for preprocessing, feature selection, hyperparameter tuning, or threshold selection.
- Preserve the original `Time` column.
- Do not randomly shuffle data in future temporal experiments.
- Treat PR-AUC as the primary metric because the target class is extremely imbalanced.
- Also calculate ROC-AUC, precision, recall, F1, specificity, and false-positive rate (FPR). Accuracy is secondary.
- Use reproducible random seeds and record them in saved experiment configurations.
- Save the actual experiment configuration, actual results, and predictions where practical.
- Never claim quantum advantage without appropriate experimental evidence.
- Clearly distinguish simulated quantum noise from results obtained on real quantum hardware.
- Use current, non-deprecated Qiskit APIs; do not use deprecated `QuantumInstance` or `qiskit.opflow` workflows.
- Keep reusable Python logic in `src/`; use notebooks only for experiments, visualization, and explanation.
- Do not duplicate important logic across notebooks.
- Do not automatically proceed to later research phases.

## Scope control

The midterm scope is limited to dataset validation, EDA, leakage-free preprocessing, Logistic Regression, RBF-SVM, XGBoost, a small quantum feature representation, preliminary QSVM/QSVC, model comparison, and midterm figures/tables.

Temporal distribution shift, simulated noise, 4/6/8-qubit scaling, computational resource analysis, explainability, optional VQC, and Streamlit are post-midterm work. Implement them only after an explicit request.
