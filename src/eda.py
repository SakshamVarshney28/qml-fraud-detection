"""Reproducible, read-only exploratory data analysis for the fraud dataset.

The functions in this module only read the source CSV and derive figures and
metadata from it. They do not preprocess, resample, remove rows, or train
models. In particular, the original ``Time`` column is retained throughout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib

# This project generates saved, reproducible PNGs rather than GUI windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data import load_dataset


NON_FRAUD_COLOR = "#4C78A8"
FRAUD_COLOR = "#E45756"
SELECTED_FEATURES = ("V1", "V2", "V3", "V4")
TIME_BIN_COUNT = 24


def _number(value: Any) -> int | float:
    """Return a JSON-serialisable Python number."""

    numeric_value = float(value)
    return int(numeric_value) if numeric_value.is_integer() else numeric_value


def _amount_statistics(amounts: pd.Series) -> Dict[str, int | float]:
    """Calculate descriptive statistics without modifying the source series."""

    description = amounts.describe()
    return {
        "count": _number(description["count"]),
        "mean": _number(description["mean"]),
        "std": _number(description["std"]),
        "min": _number(description["min"]),
        "25%": _number(description["25%"]),
        "50%": _number(description["50%"]),
        "75%": _number(description["75%"]),
        "max": _number(description["max"]),
    }


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Save and close a figure with consistent, publication-readable settings."""

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_class_distribution(dataset: pd.DataFrame, output_path: Path) -> None:
    """Plot observed class counts on a log scale to show both classes."""

    counts = dataset["Class"].value_counts().reindex([0, 1], fill_value=0)
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        ["Non-fraud (0)", "Fraud (1)"],
        counts.to_list(),
        color=[NON_FRAUD_COLOR, FRAUD_COLOR],
    )
    axis.set_yscale("log")
    axis.set_xlabel("Transaction class")
    axis.set_ylabel("Transaction count (log scale)")
    axis.set_title("Class Distribution")
    for bar, count in zip(bars, counts.values):
        axis.annotate(
            f"{int(count):,}",
            (bar.get_x() + bar.get_width() / 2, count),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )
    _save_figure(figure, output_path)


def plot_fraud_percentage(dataset: pd.DataFrame, output_path: Path) -> None:
    """Plot the observed percentage composition of transaction classes."""

    percentages = (
        dataset["Class"].value_counts(normalize=True).reindex([0, 1], fill_value=0) * 100
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        ["Non-fraud (0)", "Fraud (1)"],
        percentages.to_list(),
        color=[NON_FRAUD_COLOR, FRAUD_COLOR],
    )
    axis.set_ylim(0, 100)
    axis.set_xlabel("Transaction class")
    axis.set_ylabel("Transactions (%)")
    axis.set_title("Fraud Percentage in the Dataset")
    for bar, percentage in zip(bars, percentages.values):
        axis.annotate(
            f"{percentage:.3f}%",
            (bar.get_x() + bar.get_width() / 2, percentage),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )
    _save_figure(figure, output_path)


def plot_amount_distribution(dataset: pd.DataFrame, output_path: Path) -> None:
    """Plot the complete, unmodified transaction-amount distribution."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(dataset["Amount"], bins=100, color=NON_FRAUD_COLOR, edgecolor="white")
    axis.set_yscale("log")
    axis.set_xlabel("Transaction amount")
    axis.set_ylabel("Transaction count (log scale)")
    axis.set_title("Transaction Amount Distribution")
    _save_figure(figure, output_path)


def plot_time_distribution(dataset: pd.DataFrame, output_path: Path) -> None:
    """Plot transaction frequency across the original Time values."""

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(dataset["Time"], bins=72, color=NON_FRAUD_COLOR, edgecolor="white")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Transaction count")
    axis.set_title("Transaction Time Distribution")
    _save_figure(figure, output_path)


def plot_fraud_transactions_over_time(dataset: pd.DataFrame, output_path: Path) -> None:
    """Plot fraud counts across equally spaced original-Time bins."""

    fraud_times = dataset.loc[dataset["Class"] == 1, "Time"]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(fraud_times, bins=48, color=FRAUD_COLOR, edgecolor="white")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Fraud transaction count")
    axis.set_title("Fraud Transactions Over Time")
    _save_figure(figure, output_path)


def plot_amount_by_class(dataset: pd.DataFrame, output_path: Path) -> None:
    """Compare full amount distributions by class without clipping observations."""

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=dataset,
        x="Class",
        y="Amount",
        hue="Class",
        palette={0: NON_FRAUD_COLOR, 1: FRAUD_COLOR},
        legend=False,
        ax=axis,  # type: ignore[arg-type]
    )
    axis.set_xticks([0, 1], ["Non-fraud (0)", "Fraud (1)"])
    axis.set_yscale("symlog", linthresh=1)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Transaction class")
    axis.set_ylabel("Transaction amount (symmetric log scale)")
    axis.set_title("Fraud vs Non-fraud Amount Distributions")
    _save_figure(figure, output_path)


def plot_selected_feature_distributions(
    dataset: pd.DataFrame, output_path: Path, features: Iterable[str] = SELECTED_FEATURES
) -> None:
    """Compare class-conditional densities for a declared, non-selected feature set."""

    feature_names = tuple(features)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, feature in zip(axes.flat, feature_names):
        for class_value, label, color in (
            (0, "Non-fraud (0)", NON_FRAUD_COLOR),
            (1, "Fraud (1)", FRAUD_COLOR),
        ):
            axis.hist(
                dataset.loc[dataset["Class"] == class_value, feature],
                bins=60,
                density=True,
                histtype="step",
                linewidth=1.5,
                label=label,
                color=color,
            )
        axis.set_title(f"{feature} Distribution by Class")
        axis.set_xlabel(feature)
        axis.set_ylabel("Density")
        axis.legend()
    _save_figure(figure, output_path)


def plot_correlation_heatmap(dataset: pd.DataFrame, output_path: Path) -> None:
    """Plot Pearson correlations for all numeric dataset columns."""

    correlations = dataset.corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(20, 16))
    sns.heatmap(
        correlations,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"label": "Pearson correlation"},
        ax=axis,  # type: ignore[arg-type]
    )
    axis.set_title("Pearson Correlation Heatmap: All Numeric Features")
    _save_figure(figure, output_path)


def _time_bin_table(dataset: pd.DataFrame, bin_count: int = TIME_BIN_COUNT) -> pd.DataFrame:
    """Aggregate transaction and fraud counts across equal-width Time bins."""

    time_edges = np.linspace(dataset["Time"].min(), dataset["Time"].max(), bin_count + 1)
    time_bins = pd.cut(dataset["Time"], bins=time_edges.tolist(), include_lowest=True)
    grouped = dataset.groupby(time_bins, observed=False)["Class"].agg(
        transaction_count="size", fraud_count="sum"
    )
    grouped["fraud_rate_percentage"] = (
        grouped["fraud_count"] / grouped["transaction_count"] * 100
    )
    grouped["time_bin_start"] = [interval.left for interval in grouped.index]
    grouped["time_bin_end"] = [interval.right for interval in grouped.index]
    grouped["time_bin_midpoint"] = (
        grouped["time_bin_start"] + grouped["time_bin_end"]
    ) / 2
    return grouped.reset_index(drop=True)


def plot_fraud_rate_by_time_bin(dataset: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Plot fraud rate across equal-width Time bins, retaining the Time column."""

    time_bins = _time_bin_table(dataset)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        time_bins["time_bin_midpoint"],
        time_bins["fraud_rate_percentage"],
        color=FRAUD_COLOR,
        marker="o",
        linewidth=1.5,
    )
    axis.set_xlabel("Time-bin midpoint (seconds)")
    axis.set_ylabel("Fraud rate (%)")
    axis.set_title(f"Fraud Rate Across {TIME_BIN_COUNT} Equal-width Time Bins")
    axis.grid(axis="y", alpha=0.3)
    _save_figure(figure, output_path)
    return time_bins


def _class_amount_statistics(dataset: pd.DataFrame) -> Dict[str, Dict[str, int | float]]:
    """Return amount statistics separately for the two observed transaction classes."""

    return {
        str(class_value): _amount_statistics(dataset.loc[dataset["Class"] == class_value, "Amount"])
        for class_value in (0, 1)
    }


def generate_eda_figures(dataset: pd.DataFrame, output_directory: str | Path) -> tuple[list[str], pd.DataFrame]:
    """Generate every required EDA figure from a loaded, unchanged dataset."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    figure_definitions = (
        ("01_class_distribution.png", plot_class_distribution),
        ("02_fraud_percentage.png", plot_fraud_percentage),
        ("03_amount_distribution.png", plot_amount_distribution),
        ("04_time_distribution.png", plot_time_distribution),
        ("05_fraud_transactions_over_time.png", plot_fraud_transactions_over_time),
        ("06_amount_distribution_by_class.png", plot_amount_by_class),
        ("07_selected_feature_distributions.png", plot_selected_feature_distributions),
        ("08_correlation_heatmap.png", plot_correlation_heatmap),
    )
    saved_figure_names: list[str] = []
    for filename, plot_function in figure_definitions:
        plot_function(dataset, output_path / filename)
        saved_figure_names.append(filename)

    fraud_rate_filename = "09_fraud_rate_by_time_bin.png"
    time_bins = plot_fraud_rate_by_time_bin(dataset, output_path / fraud_rate_filename)
    saved_figure_names.append(fraud_rate_filename)
    return saved_figure_names, time_bins


def summarize_eda(
    dataset: pd.DataFrame, time_bins: pd.DataFrame, figure_names: list[str], dataset_path: str | Path
) -> Dict[str, Any]:
    """Return numerically supported EDA observations for transparent reporting."""

    class_counts = dataset["Class"].value_counts().reindex([0, 1], fill_value=0)
    fraud_rate = class_counts.loc[1] / len(dataset) * 100
    peak_fraud_count_bin = time_bins.loc[time_bins["fraud_count"].idxmax()]
    peak_fraud_rate_bin = time_bins.loc[time_bins["fraud_rate_percentage"].idxmax()]

    def bin_details(bin_row: pd.Series) -> Dict[str, int | float]:
        return {
            "time_bin_start": _number(bin_row["time_bin_start"]),
            "time_bin_end": _number(bin_row["time_bin_end"]),
            "transaction_count": _number(bin_row["transaction_count"]),
            "fraud_count": _number(bin_row["fraud_count"]),
            "fraud_rate_percentage": _number(bin_row["fraud_rate_percentage"]),
        }

    return {
        "dataset_path": str(dataset_path),
        "row_count": len(dataset),
        "class_distribution": {str(value): class_counts.loc[value] for value in (0, 1)},
        "fraud_percentage": _number(fraud_rate),
        "amount_statistics_overall": _amount_statistics(dataset["Amount"]),
        "amount_statistics_by_class": _class_amount_statistics(dataset),
        "time_min": _number(dataset["Time"].min()),
        "time_max": _number(dataset["Time"].max()),
        "time_is_monotonically_increasing": dataset["Time"].is_monotonic_increasing,
        "fraud_rate_time_bin_count": TIME_BIN_COUNT,
        "time_bin_with_most_fraud_transactions": bin_details(peak_fraud_count_bin),  # type: ignore[arg-type]
        "time_bin_with_highest_fraud_rate": bin_details(peak_fraud_rate_bin),  # type: ignore[arg-type]
        "generated_figures": figure_names,
        "selected_feature_distribution_plots": list(SELECTED_FEATURES),
    }


def save_eda_summary(summary: Dict[str, Any], output_path: str | Path) -> None:
    """Save actual EDA metadata as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2)
        file_handle.write("\n")


def run_eda(
    dataset_path: str | Path,
    figures_directory: str | Path,
    summary_path: str | Path,
) -> Dict[str, Any]:
    """Load the real CSV once, generate figures, and save calculated metadata."""

    dataset = load_dataset(dataset_path)
    figure_names, time_bins = generate_eda_figures(dataset, figures_directory)
    summary = summarize_eda(dataset, time_bins, figure_names, dataset_path)
    save_eda_summary(summary, summary_path)
    return summary


def main() -> None:
    """Run the EDA pipeline from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to creditcard.csv")
    parser.add_argument("--figures-dir", type=Path, required=True, help="Directory for PNG figures")
    parser.add_argument("--summary", type=Path, required=True, help="Path for EDA JSON metadata")
    arguments = parser.parse_args()

    summary = run_eda(arguments.dataset, arguments.figures_dir, arguments.summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
