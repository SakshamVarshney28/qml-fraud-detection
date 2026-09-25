"""Leakage checks using the real local fraud dataset only."""

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from data import load_dataset  # noqa: E402
from preprocessing import (  # noqa: E402
    SplitConfig,
    create_baseline_split,
    fit_preprocessor,
    transform_features,
)


class PreprocessingLeakageTests(unittest.TestCase):
    """Check that the scaler learns statistics from training features only."""

    @classmethod
    def setUpClass(cls) -> None:
        dataset = load_dataset(PROJECT_ROOT / "data/raw/creditcard.csv")
        cls.split = create_baseline_split(dataset, config=SplitConfig(random_state=42))
        cls.preprocessor = fit_preprocessor(cls.split.X_train, cls.split.feature_columns)

    def test_time_is_retained_for_each_partition(self) -> None:
        self.assertIn("Time", self.split.feature_columns)
        self.assertTrue(self.split.time_train.equals(self.split.X_train["Time"]))
        self.assertTrue(self.split.time_test.equals(self.split.X_test["Time"]))

    def test_scaler_statistics_match_training_features_not_test_features(self) -> None:
        scaler = self.preprocessor.named_transformers_["numeric"]
        training_means = self.split.X_train.loc[:, self.split.feature_columns].mean().to_numpy()
        test_means = self.split.X_test.loc[:, self.split.feature_columns].mean().to_numpy()

        np.testing.assert_allclose(scaler.mean_, training_means)
        self.assertFalse(np.allclose(scaler.mean_, test_means))

    def test_test_features_can_only_be_transformed_with_the_training_fitted_scaler(self) -> None:
        transformed_test = transform_features(self.preprocessor, self.split.X_test)
        self.assertEqual(transformed_test.shape, self.split.X_test.shape)
        self.assertEqual(list(transformed_test.columns), list(self.split.feature_columns))
