# Fraudulent Job Posting Detection
![tests](https://github.com/divyayan-das/Fraudulent_Job_Posting_Detection/actions/workflows/tests.yml/badge.svg)

Machine learning system that flags fraudulent job advertisements using NLP and metadata features, trained on 17,880 real postings, deployed as a live web app.

**🔴 Live Demo:** [fraud-posting-detector.streamlit.app](https://fraud-posting-detector.streamlit.app/)

## The Problem

Online job boards are flooded with fake postings designed to harvest personal information, bank details, or "registration fees" from job seekers. With thousands of ads posted daily, manual moderation can't keep up, this project automates the screening as a binary classification task.

## Approach

- **Text features:** title, company profile, description, requirements, and benefits are cleaned (lowercasing, URL/punctuation stripping) and combined, then vectorized with **TF-IDF** (12,000 features)
- **Metadata features:** 11 structured fields (employment type, has_company_logo, telecommuting, required experience/education, etc.), **one-hot encoded**, since fraud often hides in what's *missing* (no logo, no company profile)
- **Class imbalance:** only ~5% of postings are fraudulent, so models train with `class_weight='balanced'` and are judged on F1/recall for the fraud class, not accuracy, which is misleading at this imbalance

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes (text only) | 0.968 | 1.000 | 0.335 | 0.502 |
| Logistic Regression (text + metadata) | 0.978 | 0.719 | 0.902 | 0.800 |
| **SGD Classifier (text + metadata)** | **0.979** | **0.721** | **0.913** | **0.806** |

Two takeaways: adding metadata to text features dramatically improves fraud recall (0.335 → 0.913), and the deployed SGD model deliberately trades some precision for recall; a missed scam costs a job seeker more than a false alarm.

## Dataset

[Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) (EMSCAD corpus): 17,880 postings, 866 fraudulent.

## Repository Structure
├── app.py              # Streamlit web app (live demo)<br>
├── tests/              # Pytest suite (unit, artifact contract, data validation)<br>
├── .github/workflows/  # CI: runs tests on every push<br>
├── model.joblib        # Serialized best pipeline (TF-IDF + one-hot + SGD)<br>
├── requirements.txt<br>
├── src/                # Training pipeline<br>
├── data/               # Dataset<br>
├── results/            # Model comparison metrics<br>
└── paper/              # IEEE-format manuscript<br>
 
## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py          # launch the web app
python src/model.py           # retrain from scratch
```

## Testing

The project includes an automated pytest suite (26 tests) covering three layers:

- **Unit tests** — text preprocessing edge cases (NaN handling, URL stripping, symbol-only input, idempotency)
- **Artifact contract tests** — the deployed `model.joblib` is tested as a black box: valid binary output, probabilities summing to 1, deterministic predictions, and robustness to unseen categories and empty text
- **Data validation** — dataset schema, binary target integrity, and expected class imbalance

A sanity test also verifies the model assigns higher fraud probability to an obviously scammy posting than to a legitimate one.

```bash
pip install pytest
pytest                        # run the full suite
pytest --junitxml=report.xml  # JUnit-XML report (used in CI)
```

Tests run automatically on every push via GitHub Actions.

## Paper

An IEEE-format manuscript describing the methodology and results is in [`paper/`](paper/). Co-authored with Rajveer Singh Sidhu and Jatin R Nair.

## Author

Divyayan Das — [GitHub](https://github.com/divyayan-das) · [LinkedIn](https://linkedin.com/in/div-das)
