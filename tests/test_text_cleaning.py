"""Unit tests for the clean_text preprocessing function."""
import numpy as np
import pytest


class TestCleanText:
    def test_lowercases_input(self, clean_text):
        assert clean_text("Software ENGINEER") == "software engineer"

    def test_removes_urls(self, clean_text):
        result = clean_text("apply now at http://totally-legit-jobs.xyz/apply today")
        assert "http" not in result
        assert "xyz" not in result

    def test_removes_https_urls(self, clean_text):
        assert "https" not in clean_text("visit https://example.com for details")

    def test_strips_punctuation_and_symbols(self, clean_text):
        result = clean_text("Earn $$$ fast!!! (no experience?)")
        assert result == "earn fast no experience"

    def test_preserves_digits(self, clean_text):
        assert clean_text("3+ years, salary 90000") == "3 years salary 90000"

    def test_collapses_whitespace(self, clean_text):
        assert clean_text("too    many\t\tspaces\n\nhere") == "too many spaces here"

    def test_strips_leading_trailing_whitespace(self, clean_text):
        assert clean_text("  padded  ") == "padded"

    def test_handles_nan(self, clean_text):
        assert clean_text(np.nan) == ""

    def test_handles_none(self, clean_text):
        assert clean_text(None) == ""

    def test_handles_empty_string(self, clean_text):
        assert clean_text("") == ""

    def test_handles_non_string_input(self, clean_text):
        assert clean_text(12345) == "12345"

    def test_output_is_idempotent(self, clean_text):
        """Cleaning already-clean text should change nothing."""
        once = clean_text("Senior Data Analyst, remote OK!")
        assert clean_text(once) == once

    @pytest.mark.parametrize("text", [
        "!!!", "???", "@#$%^&*", "   ", "\n\t",
    ])
    def test_symbol_only_input_returns_empty(self, clean_text, text):
        assert clean_text(text) == ""
