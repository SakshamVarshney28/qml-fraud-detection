"""Leakage-free baseline splitting and scaling for the midterm experiments.

This module creates the midterm's conventional stratified random train/test
split. It deliberately keeps the raw ``Time`` values with each split so a later
chronological experiment can use them. No model is trained here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import load_dataset


TARGET_COLUMN = "Class"
TIME_COLUMN = "Time"


@dataclass(frozen=True)
class SplitConfig:
    """Explicit settings for the midterm conventional evaluation split."""

    test_size: float = 0.20
    random_state: int = 42
    shuffle: bool = True
    stratify_target: bool = True
    strategy: str = "stratified_random_baseline"


@dataclass
class DatasetSplit:
    """Raw feature and target partitions, including original Time values."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    time_train: pd.Series
    time_test: pd.Series
    feature_columns: tuple[str, ...]
    target_column: str
    time_column: str
    config: SplitConfig


@dataclass
class PreparedDataset:
    """Training-fitted transformed features plus untouched raw split metadata."""

    split: DatasetSplit
    preprocessor: ColumnTransformer
    X_train_transformed: pd.DataFrame
    X_test_transformed: pd.DataFrame


def get_feature_columns(
    dataset: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    time_column: str = TIME_COLUMN,
) -> tuple[str, ...]:
    """Return every non-target column; this is not feature selection.

    ``Time`` remains among the returned baseline features and is also preserved
    separately in :class:`DatasetSplit` for later chronological experiments.
    """

    required_columns = {target_column, time_column, "Amount"}
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return tuple(column for column in dataset.columns if column != target_column)


def create_baseline_split(
    dataset: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
    target_column: str = TARGET_COLUMN,
    time_column: str = TIME_COLUMN,
) -> DatasetSplit:
    """Create a reproducible stratified random split for the midterm only.

    This is intentionally not the protocol for the future temporal study. The
    chronological study must create a separate, non-shuffled split using the
    original ``Time`` values retained here.
    """

    if not config.shuffle:
        raise ValueError("The conventional baseline split requires shuffle=True.")
    if not config.stratify_target:
        raise ValueError("The conventional baseline split requires target stratification.")
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    feature_columns = get_feature_columns(dataset, target_column, time_column)
    train_indices, test_indices = train_test_split(
        dataset.index,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=config.shuffle,
        stratify=dataset[target_column],
    )

    # Copies protect the in-memory source dataframe from any later operations.
    X_train = dataset.loc[train_indices, feature_columns].copy()
    X_test = dataset.loc[test_indices, feature_columns].copy()
    y_train = dataset.loc[train_indices, target_column].copy()
    y_test = dataset.loc[test_indices, target_column].copy()
    time_train = dataset.loc[train_indices, time_column].copy()
    time_test = dataset.loc[test_indices, time_column].copy()

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        time_train=time_train,
        time_test=time_test,
        feature_columns=feature_columns,
        target_column=target_column,
        time_column=time_column,
        config=config,
    )


def build_preprocessor(feature_columns: Sequence[str]) -> ColumnTransformer:
    """Build a shared numerical scaler without fitting it.

    All baseline input columns, including ``Time`` and ``Amount``, use one
    ``StandardScaler``. The scaler is fitted only by :func:`fit_preprocessor`
    on training features. No labels are accepted by this function or passed to
    the scaler.
    """

    preprocessor = ColumnTransformer(
        transformers=[("numeric", StandardScaler(), list(feature_columns))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")  # type: ignore[return-value]


def fit_preprocessor(
    X_train: pd.DataFrame, feature_columns: Sequence[str]
) -> ColumnTransformer:
    """Fit the scaler exclusively on raw training features, never labels/test data."""

    missing_columns = sorted(set(feature_columns).difference(X_train.columns))
    if missing_columns:
        raise ValueError(f"Training features are missing columns: {missing_columns}")
    preprocessor = build_preprocessor(feature_columns)
    return preprocessor.fit(X_train)


def transform_features(preprocessor: ColumnTransformer, features: pd.DataFrame) -> pd.DataFrame:
    """Transform features with an already training-fitted preprocessor."""

    transformed = preprocessor.transform(features)
    if not isinstance(transformed, pd.DataFrame):
        raise TypeError("Preprocessor output must be a pandas DataFrame.")
    return transformed


def prepare_baseline_data(
    dataset: pd.DataFrame, config: SplitConfig = SplitConfig()
) -> PreparedDataset:
    """Create the split, fit on training features, and transform both partitions."""

    split = create_baseline_split(dataset, config=config)
    preprocessor = fit_preprocessor(split.X_train, split.feature_columns)
    return PreparedDataset(
        split=split,
        preprocessor=preprocessor,
        X_train_transformed=transform_features(preprocessor, split.X_train),
        X_test_transformed=transform_features(preprocessor, split.X_test),
    )


def _class_counts(target: pd.Series) -> Dict[str, int]:
    """Return JSON-safe target counts for split reporting only."""

    return {str(class_value): count for class_value, count in target.value_counts().sort_index().items()}


def preprocessing_configuration(prepared: PreparedDataset, dataset_path: str | Path) -> Dict[str, Any]:
    """Create a transparent record of the actual split and preprocessing setup."""

    split = prepared.split
    return {
        "dataset_path": str(dataset_path),
        "target_column": split.target_column,
        "time_column": split.time_column,
        "feature_columns": list(split.feature_columns),
        "feature_count": len(split.feature_columns),
        "amount_handling": {
            "column": "Amount",
            "method": "StandardScaler with all numerical baseline features",
            "fit_scope": "training features only",
        },
        "scaling": {
            "transformer": "StandardScaler",
            "columns": list(split.feature_columns),
            "fit_scope": "X_train only",
            "uses_target_labels": False,
        },
        "split": {
            "strategy": split.config.strategy,
            "test_size": split.config.test_size,
            "random_state": split.config.random_state,
            "shuffle": split.config.shuffle,
            "stratify_target": split.config.stratify_target,
            "train_rows": len(split.X_train),
            "test_rows": len(split.X_test),
            "train_class_distribution": _class_counts(split.y_train),
            "test_class_distribution": _class_counts(split.y_test),
        },
        "time_preservation": {
            "original_time_retained_in_raw_train_and_test": True,
            "future_temporal_protocol": "Use a separate chronological, non-shuffled split.",
        },
        "leakage_prevention": [
            "The split is created before scaler fitting.",
            "StandardScaler is fitted only on X_train.",
            "X_test is transformed only after fitting; it is never passed to fit.",
            "Neither y_train nor y_test is passed to preprocessing.",
            "All non-target columns are retained; no feature selection is performed.",
        ],
    }


def save_preprocessing_configuration(configuration: Dict[str, Any], output_path: str | Path) -> None:
    """Save the actual preprocessing configuration as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(configuration, file_handle, indent=2)
        file_handle.write("\n")


def run_baseline_preprocessing(
    dataset_path: str | Path,
    configuration_output_path: str | Path,
    config: SplitConfig = SplitConfig(),
) -> PreparedDataset:
    """Run the midterm preprocessing pipeline and save its configuration only."""

    dataset = load_dataset(dataset_path)
    prepared = prepare_baseline_data(dataset, config=config)
    configuration = preprocessing_configuration(prepared, dataset_path)
    save_preprocessing_configuration(configuration, configuration_output_path)
    return prepared


def main() -> None:
    """Run leakage-free baseline preprocessing without training any model."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to creditcard.csv")
    parser.add_argument("--output", type=Path, required=True, help="JSON configuration output")
    arguments = parser.parse_args()

    prepared = run_baseline_preprocessing(arguments.dataset, arguments.output)
    configuration = preprocessing_configuration(prepared, arguments.dataset)
    print(json.dumps(configuration, indent=2))


if __name__ == "__main__":
    main()
