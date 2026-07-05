
import ast
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_SOURCE = ROOT / "src" / "model.py"      
MODEL_ARTIFACT = ROOT / "model.joblib"
DATA_CSV = ROOT / "data" / "fake_job_postings.csv"

META_COLS = [
    "location", "department", "salary_range",
    "telecommuting", "has_company_logo", "has_questions",
    "employment_type", "required_experience", "required_education",
    "industry", "function",
]


@pytest.fixture(scope="session")
def clean_text():
    """Extract the clean_text function from model.py without importing it
    (importing model.py would trigger a full training run)."""
    source = MODEL_SOURCE.read_text()
    tree = ast.parse(source)
    func_node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "clean_text"
    )
    namespace = {}
    exec(  # noqa: S102 - executing our own repo's function definition only
        compile(ast.Module(body=[func_node], type_ignores=[]), str(MODEL_SOURCE), "exec"),
        {"pd": pd, "re": __import__("re")},
        namespace,
    )
    return namespace["clean_text"]


@pytest.fixture(scope="session")
def model():
    """The deployed pipeline artifact, loaded once per test session."""
    if not MODEL_ARTIFACT.exists():
        pytest.skip("model.joblib not found - run src/model.py to train first")
    return joblib.load(MODEL_ARTIFACT)


@pytest.fixture(scope="session")
def dataset():
    if not DATA_CSV.exists():
        pytest.skip("dataset CSV not found")
    return pd.read_csv(DATA_CSV)


def make_posting(**overrides) -> pd.DataFrame:
    """One model-ready input row (mirrors the columns the pipeline expects)."""
    base = {
        "combined_text": (
            "software engineer established technology firm develop and maintain "
            "backend services in python 3 years experience required health "
            "insurance and paid leave"
        ),
        "location": "US, NY, New York",
        "department": "Engineering",
        "salary_range": "90000-120000",
        "telecommuting": "0",
        "has_company_logo": "1",
        "has_questions": "1",
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree",
        "industry": "Information Technology and Services",
        "function": "Engineering",
    }
    base.update(overrides)
    return pd.DataFrame([base])


@pytest.fixture
def legit_posting():
    return make_posting()


@pytest.fixture
def scam_posting():
    return make_posting(
        combined_text=(
            "earn money from home no experience needed urgent hiring pay small "
            "registration fee earn 5000 weekly immediate start wire transfer "
            "work from home guaranteed income"
        ),
        location="missing",
        department="missing",
        salary_range="missing",
        telecommuting="1",
        has_company_logo="0",
        has_questions="0",
        employment_type="missing",
        required_experience="missing",
        required_education="missing",
        industry="missing",
        function="missing",
    )
