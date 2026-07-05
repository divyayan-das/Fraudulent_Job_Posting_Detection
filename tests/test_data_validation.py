"""Validation tests for data/fake_job_postings.csv.

Guards against silent schema drift if the dataset is ever replaced or updated.
"""
import pytest

from tests.conftest import META_COLS

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
TARGET = "fraudulent"


class TestDatasetSchema:
    def test_required_columns_present(self, dataset):
        missing = set(TEXT_COLS + META_COLS + [TARGET]) - set(dataset.columns)
        assert not missing, f"Dataset missing expected columns: {missing}"

    def test_dataset_not_empty(self, dataset):
        assert len(dataset) > 0

    def test_target_is_strictly_binary(self, dataset):
        assert set(dataset[TARGET].dropna().unique()) <= {0, 1}

    def test_target_has_no_missing_values(self, dataset):
        assert dataset[TARGET].isna().sum() == 0

    def test_both_classes_present(self, dataset):
        """Stratified split and class_weight='balanced' both require 2 classes."""
        assert dataset[TARGET].nunique() == 2

    def test_no_duplicate_job_ids(self, dataset):
        if "job_id" in dataset.columns:
            assert dataset["job_id"].is_unique
        else:
            pytest.skip("no job_id column in dataset")

    def test_class_imbalance_is_expected_direction(self, dataset):
        """Fraudulent postings should be the minority class; if not,
        class_weight and evaluation assumptions need revisiting."""
        fraud_rate = dataset[TARGET].mean()
        assert 0 < fraud_rate < 0.5
