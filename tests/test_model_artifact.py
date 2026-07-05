"""Black-box contract tests for the deployed model.joblib pipeline.

These verify the artifact the Streamlit app actually serves: it loads,
accepts the expected input schema, and produces sane, stable outputs.
"""
import numpy as np
import pandas as pd
import pytest

from tests.conftest import META_COLS, make_posting


class TestArtifactContract:
    def test_artifact_loads_and_exposes_api(self, model):
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_prediction_is_binary(self, model, legit_posting):
        pred = model.predict(legit_posting)
        assert pred.shape == (1,)
        assert pred[0] in (0, 1)

    def test_probabilities_are_valid(self, model, legit_posting):
        proba = model.predict_proba(legit_posting)
        assert proba.shape == (1, 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert proba.sum(axis=1) == pytest.approx(1.0)

    def test_batch_input_returns_matching_output(self, model):
        batch = pd.concat([make_posting() for _ in range(5)], ignore_index=True)
        preds = model.predict(batch)
        assert len(preds) == 5

    def test_predictions_are_deterministic(self, model, legit_posting):
        """Same input, same output - required for a deployed inference service."""
        p1 = model.predict_proba(legit_posting)
        p2 = model.predict_proba(legit_posting)
        np.testing.assert_array_equal(p1, p2)


class TestRobustness:
    def test_handles_empty_text(self, model):
        posting = make_posting(combined_text="")
        model.predict(posting)  # must not raise

    def test_handles_unseen_categories(self, model):
        """OneHotEncoder was built with handle_unknown='ignore'; verify it holds."""
        posting = make_posting(
            location="ZZ, Nowhere", industry="Underwater Basket Weaving",
            employment_type="Quantum Part-Time",
        )
        model.predict(posting)  # must not raise

    def test_handles_missing_metadata_sentinel(self, model):
        posting = make_posting(**{col: "missing" for col in META_COLS})
        model.predict(posting)  # must not raise

    def test_handles_very_long_text(self, model):
        posting = make_posting(combined_text="great opportunity " * 5000)
        model.predict(posting)  # must not raise


class TestModelSanity:
    def test_scam_scores_higher_fraud_probability_than_legit(
        self, model, legit_posting, scam_posting
    ):
        p_legit = model.predict_proba(legit_posting)[0, 1]
        p_scam = model.predict_proba(scam_posting)[0, 1]
        assert p_scam > p_legit, (
            f"Expected scam posting (p={p_scam:.3f}) to score higher fraud "
            f"probability than legit posting (p={p_legit:.3f})"
        )

    def test_legit_posting_classified_as_legit(self, model, legit_posting):
        assert model.predict(legit_posting)[0] == 0
