# Project Status and Handoff

## Project objective

**Title:** *Noise-Aware Hybrid Quantum-Classical Financial Fraud Detection Under Class Imbalance and Temporal Distribution Shift*

This repository is a university research project using the ULB Credit Card Fraud Detection dataset. Its final objective is to compare classical baselines with a quantum-kernel QSVM/QSVC while addressing extreme class imbalance, time-related distribution concerns, simulated quantum noise, qubit/feature scaling, and computational cost. No claim of quantum advantage is currently supported or permitted.

## Research questions

The intended final research questions are:

1. How severe class imbalance affects fraud detection evaluation.
2. How classical methods compare with a quantum-kernel method.
3. How chronological/temporal distribution shift affects results.
4. How simulated quantum noise affects quantum-model results.
5. How quantum feature/qubit scaling affects performance.
6. What computational cost each approach incurs.

## Dataset

- Dataset: ULB Credit Card Fraud Detection dataset.
- Exact local path: `data/raw/creditcard.csv`.
- The raw CSV exists locally and is ignored by Git. Do not modify it.
- Validated shape: 284,807 rows and 31 columns.
- Columns: `Time`, `V1` through `V28`, `Amount`, and `Class`.
- Target: `Class`; observed values are `0` and `1`.
- Class distribution: 284,315 non-fraud rows and 492 fraud rows (0.1727485630620034% fraud).
- No missing values were found; 1,081 duplicate rows were found. No rows have been removed.
- `Time` ranges from 0 to 172,792 and is monotonically increasing in the raw CSV.

Full calculated metadata is in `results/json/dataset_summary.json`.

## Current repository structure

```text
.
├── AGENTS.md
├── PROJECT_STATUS.md
├── README.md
├── requirements.txt
├── configs/                         # Empty placeholder directory
├── data/
│   ├── raw/creditcard.csv            # Local raw dataset; do not modify
│   ├── interim/                      # Empty placeholder directory
│   └── processed/                    # Empty placeholder directory
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classical_baselines.ipynb
│   ├── 04_quantum_basics.ipynb
│   ├── 05_qsvm_baseline.ipynb
│   ├── 06_midterm_results.ipynb
│   └── 07_fraud_detection_analysis.ipynb
├── results/
│   ├── figures/eda/                  # 9 generated EDA PNGs
│   ├── figures/classical/            # 4 generated classical-model PNGs
│   ├── figures/qsvm/                 # Preliminary QSVM figures and circuit text
│   ├── figures/fraud_detection/      # Fraud-analysis confusion/count figures
│   ├── figures/midterm_model_comparison.png
│   ├── json/                         # Calculated metadata, results, predictions
│   └── tables/                       # Midterm and fraud-analysis tables
├── app/
│   └── streamlit_app.py              # Saved-artifact demonstration app
├── src/
│   ├── data.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── classical_models.py
│   ├── evaluation.py
│   ├── quantum_features.py
│   └── qsvm.py
│   └── fraud_detection/__init__.py   # Placeholder package; currently unused
└── tests/
    ├── test_project_layout.py
    └── test_preprocessing.py
```

`__pycache__/` directories are generated locally and ignored. No `.git` directory was detected during the handoff audit.

## Completed work

### Dataset validation

Completed with reusable code in `src/data.py` and notebook `notebooks/01_dataset_eda.ipynb`.

- Checks file existence/loadability, schema, dtypes, missing values, duplicate rows, class balance, time range/order, amount statistics, and binary target values.
- Saves actual output to `results/json/dataset_summary.json`.

### Exploratory data analysis

Completed with reusable code in `src/eda.py` and the same first notebook.

- Generated nine figures in `results/figures/eda/`:
  1. `01_class_distribution.png`
  2. `02_fraud_percentage.png`
  3. `03_amount_distribution.png`
  4. `04_time_distribution.png`
  5. `05_fraud_transactions_over_time.png`
  6. `06_amount_distribution_by_class.png`
  7. `07_selected_feature_distributions.png`
  8. `08_correlation_heatmap.png`
  9. `09_fraud_rate_by_time_bin.png`
- The selected distribution plot is explicitly a visualisation of `V1`, `V2`, `V3`, and `V4`; it is not feature selection.
- Saves actual numerical EDA metadata to `results/json/eda_summary.json`.

### Leakage-free baseline preprocessing

Completed with reusable code in `src/preprocessing.py` and notebook `notebooks/02_preprocessing.ipynb`.

- Defines the target as `Class`.
- Uses every non-target column as a baseline feature: `Time`, `V1`–`V28`, and `Amount` (30 features).
- Preserves the unscaled original `Time` series separately for the train and test partitions.
- Uses `StandardScaler` for every numerical baseline input, including `Time` and `Amount`.
- Fits the scaler only on `X_train`; neither `y_train` nor `y_test` is passed to preprocessing.
- Saves the actual methodology and split data to `results/json/preprocessing_config.json`.

### Classical baselines and evaluation

Completed with reusable code in `src/classical_models.py` and `src/evaluation.py`, and notebook `notebooks/03_classical_baselines.ipynb`.

- Logistic Regression, RBF-SVM, and XGBoost were run on the same prepared train/test split.
- PR-AUC is implemented as average precision and is the stated primary metric.
- ROC-AUC, precision, recall, F1, specificity, FPR, confusion matrices, training time, and inference time are calculated.
- Model predictions use native model prediction rules; no threshold was selected using test labels.
- Actual results are saved to `results/json/classical_results.json`.
- Actual shared-test predictions are saved to `results/json/classical_test_predictions.json` (56,962 records; about 17 MB).
- Four figures are saved in `results/figures/classical/`:
  1. `01_pr_curves.png`
  2. `02_roc_curves.png`
  3. `03_confusion_matrices.png`
  4. `04_pr_auc_comparison.png`

### Actual classical results

These values are copied from the existing calculated file `results/json/classical_results.json`; do not replace them with manually entered values in future code.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Specificity | FPR | Train seconds | Inference seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7189705771419241 | 0.9720834996210077 | 0.06097560975609756 | 0.9183673469387755 | 0.11435832274459974 | 0.9756260551491277 | 0.024373944850872256 | 0.16914454201469198 | 0.0015764579875394702 |
| RBF-SVM | 0.4777797521079116 | 0.9731911370344424 | 0.3217391304347826 | 0.7551020408163265 | 0.45121951219512196 | 0.9972566122678672 | 0.0027433877321328083 | 22.42762995796511 | 15.52804504201049 |
| XGBoost | 0.8888909148075554 | 0.9786806042056666 | 0.8736842105263158 | 0.8469387755102041 | 0.8601036269430051 | 0.9997889701744513 | 0.00021102982554867754 | 1.309666917019058 | 0.04075083404313773 |

Confusion matrices use `[[TN, FP], [FN, TP]]`:

- Logistic Regression: `[[55478, 1386], [8, 90]]`
- RBF-SVM: `[[56708, 156], [24, 74]]`
- XGBoost: `[[56852, 12], [15, 83]]`

The RBF-SVM configuration has `max_iter=10000`; the stored `fit_status` is `0` and `fit_warnings` is empty for the completed run.

### Quantum feature reduction and preliminary QSVM

Completed with reusable code in `src/quantum_features.py` and `src/qsvm.py`, and notebooks `notebooks/04_quantum_basics.ipynb` and `notebooks/05_qsvm_baseline.ipynb`.

- Training-only ANOVA F-score reduction selects `V17`, `V14`, `V12`, and `V10` for the four-feature representation.
- The current supported API path is `zz_feature_map` with 4 qubits, `reps=1`, linear entanglement, `StatevectorSampler`, `ComputeUncompute`, `FidelityQuantumKernel`, and `QSVC`.
- The preliminary QSVM uses deterministic 32-row training and 32-row test subsets, each with 16 non-fraud and 16 fraud rows, selected from the existing baseline partitions only.
- Saved output is in `results/json/qsvm_results.json` and `results/json/qsvm_predictions.json`.
- The 32/32 QSVM result is preliminary and not directly equivalent to the full-data classical evaluation. It does not support a quantum-advantage claim.

### Midterm comparison and fraud-detection analysis

Completed with `notebooks/06_midterm_results.ipynb` and `notebooks/07_fraud_detection_analysis.ipynb`.

- `results/tables/midterm_model_comparison.csv` and `.md` contain the factual comparison without ranking models.
- `results/figures/midterm_model_comparison.png` shows PR-AUC values while identifying the QSVM subset limitation.
- Fraud analysis uses all 56,962 saved classical test predictions without retraining or threshold optimization.
- The full classical test set contains 98 frauds and 56,864 legitimate transactions.
- Fraud detections / false positives: Logistic Regression 90 / 1,386; RBF-SVM 74 / 156; XGBoost 83 / 12.
- Analysis output is saved to `results/json/fraud_detection_analysis.json`; figures are in `results/figures/fraud_detection/`.

### Initial Streamlit demonstration

Completed in `app/streamlit_app.py`.

- Uses cached saved JSON, CSV, prediction, and figure artifacts only.
- Provides model selection and held-out transaction selection for existing classical predictions.
- Displays project features `Time`, `Amount`, `V17`, `V14`, `V12`, and `V10` as read-only values for the selected held-out transaction.
- Includes saved QSVM metrics and selection of existing QSVM test predictions.
- Does not retrain models, launch QSVM computation, or claim arbitrary new-transaction inference.
- Streamlit is included in `requirements.txt`.

## Current preprocessing and split methodology

### Conventional midterm baseline split

This split is only for the midterm classical/initial-quantum baseline. It is not the future temporal protocol.

- Strategy: `stratified_random_baseline`.
- Test size: 0.20.
- Random seed: 42.
- Shuffle: `True`.
- Stratification: `Class`.
- Training rows: 227,845 (227,451 non-fraud; 394 fraud).
- Test rows: 56,962 (56,864 non-fraud; 98 fraud).

The split is regenerated deterministically by `prepare_baseline_data()` using the raw CSV and the configuration above. No fitted scaler or trained model object is saved; rerunning with the same environment/data recreates them.

### Scaling and Amount handling

`StandardScaler` is applied to all 30 baseline features and fitted on only the raw training feature matrix. `Amount` is not removed or treated with a special transformation; it is one column in this shared scaler. `Time` is also scaled for the conventional baseline input but its raw, unscaled values remain in `DatasetSplit.time_train` and `DatasetSplit.time_test`.

## Random seeds and fixed model settings

- Shared split seed: `42`.
- Logistic Regression: `random_state=42`, `C=1.0`, `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=1000`.
- RBF-SVM: `random_state=42`, RBF kernel, `C=1.0`, `gamma="scale"`, `class_weight="balanced"`, `cache_size=1024`, `max_iter=10000`.
- XGBoost: `random_state=42`, 300 estimators, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`, and `scale_pos_weight=577.2868020304569` calculated from training labels only.

There is no hyperparameter search implementation or saved tuning process.

## Python environment and package versions

Active environment:

- Python executable: `.venv/bin/python`
- Python version: `3.12.14`
- Platform compiler reported by Python: Clang 21.0.0

Installed packages observed during this audit:

| Package | Version |
| --- | --- |
| numpy | 2.5.3 |
| pandas | 2.3.3 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| qiskit | 2.3.0 |
| qiskit-machine-learning | 0.9.1 |
| jupyterlab | 4.6.4 |
| pytest | 9.1.1 |

On this macOS environment, XGBoost required the Homebrew `libomp` runtime. It was installed locally during the classical-baseline work. `qiskit-aer` is not listed in `requirements.txt` and no Aer/noise implementation exists yet.

## Important files

| File | Actual purpose |
| --- | --- |
| `AGENTS.md` | Mandatory research rules and phase boundaries. Read this first. |
| `requirements.txt` | Intended Python dependencies; requires Python 3.10+ due to the Qiskit stack. |
| `src/data.py` | Read-only CSV loader and validation metadata generator. |
| `src/eda.py` | Read-only EDA figures and EDA-summary generator; configures Matplotlib `Agg` backend. |
| `src/preprocessing.py` | Baseline split dataclasses, training-only `StandardScaler`, transformed matrices, and JSON config writer. |
| `src/classical_models.py` | Fixed model constructors and training-label-only XGBoost class-weight calculation. |
| `src/evaluation.py` | Fitting, score extraction, required metrics, prediction JSON writer, figures, and classical experiment CLI. |
| `src/quantum_features.py` | Training-only four-feature ANOVA selector for the preliminary quantum representation. |
| `src/qsvm.py` | Current-API reduced-subset noiseless QSVC experiment and artifact writer. |
| `results/json/dataset_summary.json` | Actual validation metadata. |
| `results/json/eda_summary.json` | Actual numerical EDA summary. |
| `results/json/preprocessing_config.json` | Actual split/scaling/leakage controls. |
| `results/json/classical_results.json` | Actual baseline metrics, timings, configurations, and split record. |
| `results/json/classical_test_predictions.json` | Actual per-test-row labels, raw `Time`, predictions, and scores for all three classical models. |
| `results/json/qsvm_results.json` | Actual reduced-subset preliminary QSVM configuration, metrics, timings, and limitations. |
| `results/json/qsvm_predictions.json` | Actual predictions and scores for the 32-row QSVM test subset. |
| `results/json/fraud_detection_analysis.json` | Actual fraud counts, confusion matrices, metrics, and false-positive counts from all classical predictions. |
| `results/tables/midterm_model_comparison.csv` | Factual classical/preliminary-QSVM comparison with matching notes. |
| `app/streamlit_app.py` | Cached saved-artifact Streamlit demonstration; no training code. |

## Existing notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/01_dataset_eda.ipynb` | Calls `src/data.py` for validation and `src/eda.py` for EDA. |
| `notebooks/02_preprocessing.ipynb` | Documents and calls the reusable baseline split/scaling pipeline. |
| `notebooks/03_classical_baselines.ipynb` | Documents and calls the shared-split classical baseline runner. |
| `notebooks/04_quantum_basics.ipynb` | Explains the four-feature quantum representation without QSVM training. |
| `notebooks/05_qsvm_baseline.ipynb` | Documents the saved preliminary reduced-subset QSVM experiment. |
| `notebooks/06_midterm_results.ipynb` | Main saved-artifact midterm results demonstration. |
| `notebooks/07_fraud_detection_analysis.ipynb` | Full classical held-out fraud-detection analysis from saved predictions. |

The notebooks are thin wrappers. Important logic belongs in `src/` and should not be duplicated in notebooks.

## Existing tests

- `tests/test_project_layout.py`: verifies required repository paths exist.
- `tests/test_preprocessing.py`: uses the real local dataset and verifies:
  - `Time` is retained in both raw split partitions.
  - `StandardScaler.mean_` matches training-feature means.
  - Those learned means do not match the test-feature means.
  - Test data is transformed using the training-fitted scaler.

The latest recorded test command was:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

It passed 4 tests. There are currently no dedicated unit tests for `src/data.py`, `src/eda.py`, `src/classical_models.py`, or `src/evaluation.py`.

## Known warnings, limitations, and documentation discrepancies

- `README.md` is stale: it says that no model has been trained/no performance result exists and names baseline-model implementation as the next step. Classical models and results now exist. This handoff documents the actual state; do not treat the README statements as current until they are deliberately updated.
- The repository is not currently a Git repository (`.git` was not present during inspection), despite having a `.gitignore` file.
- Matplotlib may warn that `/Users/sakshamvarshney/.matplotlib` is not writable and create a temporary cache. Existing EDA/evaluation code uses the non-GUI `Agg` backend. Previous figure commands used `MPLCONFIGDIR=/private/tmp/qml-fraud-detection-matplotlib` to avoid this cache warning.
- XGBoost on this macOS installation initially failed because `libomp.dylib` was missing. Homebrew `libomp` has now been installed; a recreated environment will need the equivalent runtime available.
- The RBF-SVM has a fixed `max_iter=10000` resource cap. Its completed run reports convergence (`fit_status=0`) and no convergence warnings, but this setting must remain recorded if changed later.
- `src` is not a conventional installed Python package. Scripts are executed as `python src/<script>.py`; notebooks and preprocessing tests explicitly add `src/` to `sys.path`. Preserve or deliberately refactor this import convention before changing imports.
- The QSVM is only a reduced-subset preliminary experiment; no full-data quantum-kernel evaluation exists.
- No temporal splitting, simulated noise, qubit scaling, final resource study, explainability, VQC, final report, or presentation exists yet.
- The Streamlit demonstration is artifact-based and does not provide inference for arbitrary new transactions because fitted model objects are not saved.

## Data-leakage considerations

- Do not fit preprocessing on the conventional baseline test data.
- Do not use the conventional baseline test labels for feature selection, representation design, hyperparameter selection, threshold selection, or model choice.
- The current 4-feature EDA plot is not a justified or frozen quantum feature-selection procedure. A future 4-feature/4-qubit representation must be defined or selected using training data only, with an explicit validation protocol if tuning is needed.
- The current random stratified split is valid only for the stated conventional baseline. It must not be silently substituted for the future chronological experiment.
- The future temporal protocol must preserve row/time order, use a separately designed chronological non-shuffled split, fit every transformer on the temporal training period only, and reserve its test period for final evaluation.
- Existing baseline predictions contain true test labels because they are experimental output. They must not be used to choose future thresholds or tune subsequent models.

## Remaining work

### Midterm

1. Logistic Regression — **completed**.
2. RBF-SVM — **completed**.
3. XGBoost — **completed**.
4. Classical model evaluation — **completed**.
5. 4-feature/4-qubit quantum feature representation — **completed**.
6. Quantum kernel — **completed for the reduced-subset preliminary experiment**.
7. QSVM/QSVC — **completed as a reduced-subset preliminary experiment**.
8. Preliminary comparison — **completed with explicit unmatched-subset notes**.
9. Midterm figures — **completed for EDA, classical, QSVM, fraud analysis, and comparison**.
10. Midterm notebook — **completed** in `notebooks/06_midterm_results.ipynb`.
11. Fraud-detection analysis — **completed** in `notebooks/07_fraud_detection_analysis.ipynb`.
12. Initial Streamlit demonstration — **completed** with saved artifacts only.
13. Midterm presentation — **incomplete**.

### Post-midterm

1. Chronological/temporal distribution shift — **incomplete**.
2. Simulated quantum noise using Qiskit Aer — **incomplete**.
3. Quantum feature/qubit scaling — **incomplete**.
4. Computational resource analysis — **incomplete**.
5. Explainability — **incomplete**.
6. Optional VQC — **incomplete**.
7. Final analysis — **incomplete**.
8. Final report — **incomplete**.
9. Final presentation — **incomplete**.
10. Optional Streamlit demonstration — **incomplete**.

## Required research rules

- Never fabricate results, metrics, citations, dataset statistics, or experimental outcomes.
- Never hardcode experimental results.
- Never use test data for preprocessing, feature selection, hyperparameter tuning, or threshold selection.
- Preserve `Time`.
- Use PR-AUC as the primary metric; also report ROC-AUC, precision, recall, F1, specificity, and FPR.
- Do not claim quantum advantage without appropriate evidence.
- Distinguish simulated quantum noise from real quantum hardware noise.
- Use current, non-deprecated Qiskit APIs; do not use deprecated `QuantumInstance` or `qiskit.opflow` workflows.
- Keep reusable implementation in `src/` and notebooks for experiments/visualisation.
- Do not silently change the research methodology or automatically advance to later phases.

## Full planned sequence through the final project

### Midterm sequence

1. Logistic Regression — completed.
2. RBF-SVM — completed.
3. XGBoost — completed.
4. Classical model evaluation — completed.
5. 4-feature/4-qubit quantum feature representation — next incomplete step.
6. Quantum kernel.
7. QSVM/QSVC.
8. Preliminary comparison.
9. Midterm figures.
10. Midterm notebook.
11. Midterm presentation.

### Post-midterm sequence

1. Chronological/temporal distribution shift.
2. Simulated quantum noise using Qiskit Aer.
3. Quantum feature/qubit scaling.
4. Computational resource analysis.
5. Explainability.
6. Optional VQC.
7. Final analysis.
8. Final report.
9. Final presentation.
10. Optional Streamlit demonstration.

## NEXT ACTION

Do not rerun completed experiments. Prepare the midterm presentation using the saved classical, fraud-analysis, comparison, and preliminary QSVM artifacts. After the midterm, design the chronological non-shuffled evaluation protocol before implementing temporal distribution shift. Keep simulated noise, qubit scaling, resource analysis, explainability, VQC, and arbitrary-new-transaction inference out of scope until explicitly requested.
