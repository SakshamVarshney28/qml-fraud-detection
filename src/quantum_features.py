"""Training-only feature reduction for the preliminary quantum experiment.

The preliminary quantum model uses four classical features for four qubits.
This module selects those features from the training partition only, after the
existing training-fitted baseline scaler has been applied. The fitted selector
can then transform the held-out partition without inspecting its labels or
feature values during selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif


DEFAULT_FEATURE_COUNT = 4


@dataclass
class TrainingFeatureSelector:
    """Select the top features using training-set ANOVA F-scores only.

    Scores are ranked by descending F-score. Exact score ties are resolved by
    the original input-column order, making the selected representation
    deterministic for a fixed training matrix and target vector.
    """

    feature_count: int = DEFAULT_FEATURE_COUNT

    def fit(self, X_train: pd.DataFrame, y_train: Sequence[object]) -> "TrainingFeatureSelector":
        """Fit feature selection using only training features and labels."""

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if X_train.shape[1] == 0:
            raise ValueError("X_train must contain at least one feature.")
        if not 0 < self.feature_count <= X_train.shape[1]:
            raise ValueError(
                "feature_count must be between 1 and the number of training features."
            )
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must contain the same number of rows.")
        if X_train.columns.duplicated().any():
            raise ValueError("X_train must not contain duplicate feature names.")
        if not np.isfinite(X_train.to_numpy(dtype=float)).all():
            raise ValueError("X_train must contain only finite numeric values.")

        scores, _ = f_classif(X_train, np.asarray(y_train))
        finite_scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        ranked_positions = sorted(
            range(X_train.shape[1]),
            key=lambda position: (-finite_scores[position], position),
        )
        selected_positions = ranked_positions[: self.feature_count]

        self.feature_names_in_ = tuple(column for column in X_train.columns)
        self.feature_scores_ = {
            column: float(score)
            for column, score in zip(X_train.columns, scores)
        }
        self.selected_features_ = tuple(X_train.columns[position] for position in selected_positions)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return only the features selected from the fitted training data."""

        if not hasattr(self, "selected_features_"):
            raise RuntimeError("TrainingFeatureSelector must be fitted before transform.")
        missing_columns = sorted(set(self.feature_names_in_).difference(features.columns))
        if missing_columns:
            raise ValueError(f"Features are missing fitted columns: {missing_columns}")
        return features.loc[:, list(self.selected_features_)].copy()

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: Sequence[object]
    ) -> pd.DataFrame:
        """Fit on training data and return the reduced training matrix."""

        return self.fit(X_train, y_train).transform(X_train)

    @property
    def selected_feature_names(self) -> tuple[str, ...]:
        """Return the frozen selected feature names after fitting."""

        if not hasattr(self, "selected_features_"):
            raise RuntimeError("TrainingFeatureSelector must be fitted first.")
        return self.selected_features_


def reduce_to_quantum_features(
    X_train: pd.DataFrame,
    y_train: Sequence[object],
    X_test: pd.DataFrame,
    feature_count: int = DEFAULT_FEATURE_COUNT,
) -> tuple[TrainingFeatureSelector, pd.DataFrame, pd.DataFrame]:
    """Fit a selector on training data and reduce train/test matrices.

    ``X_test`` is passed only to ``transform`` after the selector is fitted;
    its values and labels cannot influence feature selection.
    """

    selector = TrainingFeatureSelector(feature_count=feature_count)
    reduced_train = selector.fit_transform(X_train, y_train)
    reduced_test = selector.transform(X_test)
    return selector, reduced_train, reduced_test
