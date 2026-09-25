"""Read-only validation utilities for the ULB credit-card fraud dataset.

This module intentionally performs no preprocessing, sampling, row removal, or
model training. It reads the CSV as stored and reports metadata calculated from
that file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


REQUIRED_COLUMNS = ("Time", "Amount", "Class")


def _as_builtin_number(value: Any) -> int | float:
    """Convert a pandas/numpy numeric scalar into a JSON-serialisable value."""

    if isinstance(value, bool):
        return int(value)
    if float(value).is_integer():
        return int(value)
    return float(value)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the dataset without changing its rows, columns, or values.

    Raises:
        FileNotFoundError: If ``csv_path`` does not point to a file.
        ValueError: If required validation columns are absent.
    """

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {path}")

    dataset = pd.read_csv(path)
    missing_required_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataset.columns
    ]
    if missing_required_columns:
        raise ValueError(
            "Dataset is missing required validation columns: "
            + ", ".join(missing_required_columns)
        )
    return dataset


def summarize_dataset(csv_path: str | Path) -> Dict[str, Any]:
    """Calculate validation metadata directly from an unchanged dataset CSV."""

    path = Path(csv_path)
    dataset = load_dataset(path)

    class_counts = dataset["Class"].value_counts(dropna=False).sort_index()
    observed_class_values = sorted(
        _as_builtin_number(value)
        for value in dataset["Class"].dropna().unique().tolist()
    )
    contains_only_binary_classes = (
        dataset["Class"].notna().all()
        and set(observed_class_values).issubset({0, 1})
    )

    fraud_count = class_counts.get(1, 0)
    row_count = dataset.shape[0]
    amount_description = dataset["Amount"].describe()

    return {
        "dataset_path": str(path),
        "file_exists": path.is_file(),
        "row_count": row_count,
        "column_count": dataset.shape[1],
        "column_names": dataset.columns.tolist(),
        "data_types": {column: str(dtype) for column, dtype in dataset.dtypes.items()},
        "missing_values_by_column": {
            column: int(count) for column, count in dataset.isna().sum().items()
        },
        "missing_values_total": int(dataset.isna().sum().sum()),
        "duplicate_row_count": int(dataset.duplicated().sum()),
        "class_distribution": {
            str(_as_builtin_number(class_value)): count
            for class_value, count in class_counts.items()
        },
        "fraud_percentage": (fraud_count / row_count * 100) if row_count else 0.0,
        "time_min": _as_builtin_number(dataset["Time"].min()),
        "time_max": _as_builtin_number(dataset["Time"].max()),
        "time_is_monotonically_increasing": dataset["Time"].is_monotonic_increasing,
        "amount_statistics": {
            "count": _as_builtin_number(amount_description["count"]),
            "mean": _as_builtin_number(amount_description["mean"]),
            "std": _as_builtin_number(amount_description["std"]),
            "min": _as_builtin_number(amount_description["min"]),
            "25%": _as_builtin_number(amount_description["25%"]),
            "50%": _as_builtin_number(amount_description["50%"]),
            "75%": _as_builtin_number(amount_description["75%"]),
            "max": _as_builtin_number(amount_description["max"]),
        },
        "class_unique_values": observed_class_values,
        "class_contains_only_0_and_1": bool(contains_only_binary_classes),
    }


def save_dataset_summary(summary: Dict[str, Any], output_path: str | Path) -> None:
    """Write calculated metadata to JSON without touching the source CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2)
        file_handle.write("\n")


def main() -> None:
    """Run read-only validation from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to creditcard.csv")
    parser.add_argument("--output", type=Path, required=True, help="Path for JSON summary")
    arguments = parser.parse_args()

    summary = summarize_dataset(arguments.dataset)
    save_dataset_summary(summary, arguments.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
