import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# CONFIG

from pathlib import Path
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "fake_job_postings.csv"
RANDOM_STATE = 42

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
META_COLS = [
    "location", "department", "salary_range",
    "telecommuting", "has_company_logo", "has_questions",
    "employment_type", "required_experience", "required_education",
    "industry", "function"
]
TARGET = "fraudulent"

# CLEAN TEXT

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# LOAD DATA

df = pd.read_csv(CSV_PATH)

# Combine text
for col in TEXT_COLS:
    df[col] = df[col].fillna("")

df["combined_text"] = df[TEXT_COLS].agg(" ".join, axis=1).apply(clean_text)

# Fill metadata
for col in META_COLS:
    df[col] = df[col].fillna("missing").astype(str)

# Target
y = df[TARGET]
X = df[["combined_text"] + META_COLS]

# SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# PIPELINES

# Naive Bayes
nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english', max_features=5000)),
    ("clf", MultinomialNB())
])

# Text + Metadata
preprocessor = ColumnTransformer([
    ("text", TfidfVectorizer(stop_words='english', max_features=12000), "combined_text"),
    ("meta", OneHotEncoder(handle_unknown='ignore'), META_COLS)
])

# Logistic Regression
lr_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, class_weight='balanced'))
])

# SGD Classifier
sgd_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", SGDClassifier(loss='log_loss', class_weight='balanced'))
])

# EVALUATION FUNCTION

def evaluate(name, model, X_test, y_test, text_only=False):
    if text_only:
        preds = model.predict(X_test["combined_text"])
    else:
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    return [name, acc, prec, rec, f1]

# TRAIN & EVALUATE

results = []

# Naive Bayes
nb_pipeline.fit(X_train["combined_text"], y_train)
results.append(evaluate("Naive Bayes (Text Only)", nb_pipeline, X_test, y_test, True))

# Logistic Regression
lr_pipeline.fit(X_train, y_train)
results.append(evaluate("Logistic Regression (Text + Metadata)", lr_pipeline, X_test, y_test))

# SGD
sgd_pipeline.fit(X_train, y_train)
results.append(evaluate("SGD Classifier (Text + Metadata)", sgd_pipeline, X_test, y_test))

# SAVE RESULTS

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

ROOT = Path(__file__).resolve().parent.parent
results_df.to_csv(ROOT / "results" / "results.csv", index=False)

joblib.dump(sgd_pipeline, ROOT / "model.joblib")

print("\nSaved results.csv and model.joblib")