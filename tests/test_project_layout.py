"""Smoke tests for the initialized repository layout.

These tests intentionally do not load data or train models.
"""

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTests(unittest.TestCase):
    """Ensure essential project locations remain available to later phases."""

    def test_required_project_paths_exist(self) -> None:
        expected_paths = (
            "AGENTS.md",
            "README.md",
            "requirements.txt",
            "configs",
            "data/raw",
            "data/interim",
            "data/processed",
            "notebooks",
            "results/figures",
            "results/json",
            "results/tables",
            "results/predictions",
            "src/fraud_detection",
        )

        for relative_path in expected_paths:
            with self.subTest(path=relative_path):
                self.assertTrue((PROJECT_ROOT / relative_path).exists())
