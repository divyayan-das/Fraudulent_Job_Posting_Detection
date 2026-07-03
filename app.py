import streamlit as st
import pandas as pd
import joblib
import re

st.set_page_config(page_title="Fraudulent Job Posting Detector", page_icon="🕵️")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

def clean_text(text):
    """Identical to training-time cleaning."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.title("🕵️ Fraudulent Job Posting Detector")
st.caption(
    "ML classifier (TF-IDF + metadata, SGD) trained on 17,880 postings "
)

with st.form("posting"):
    title = st.text_input("Job Title", placeholder="e.g., Data Entry Clerk — Work From Home")
    company_profile = st.text_area("Company Profile", height=80)
    description = st.text_area("Job Description", height=150)
    requirements = st.text_area("Requirements", height=80)
    benefits = st.text_area("Benefits", height=80)

    col1, col2, col3 = st.columns(3)
    with col1:
        employment_type = st.selectbox(
            "Employment Type",
            ["missing", "Full-time", "Part-time", "Contract", "Temporary", "Other"],
        )
        telecommuting = st.checkbox("Remote / telecommuting")
    with col2:
        required_experience = st.selectbox(
            "Required Experience",
            ["missing", "Internship", "Entry level", "Associate",
             "Mid-Senior level", "Director", "Executive", "Not Applicable"],
        )
        has_company_logo = st.checkbox("Posting has company logo", value=True)
    with col3:
        required_education = st.selectbox(
            "Required Education",
            ["missing", "High School or equivalent", "Bachelor's Degree",
             "Master's Degree", "Doctorate", "Certification", "Unspecified"],
        )
        has_questions = st.checkbox("Has screening questions")

    submitted = st.form_submit_button("Analyze Posting")

if submitted:
    combined = clean_text(
        " ".join([title or "", company_profile or "", description or "",
                  requirements or "", benefits or ""])
    )

    if len(combined.split()) < 5:
        st.warning(
            "⚠️ Please enter the job posting text first, at minimum a title and "
            "description. The model can't assess an empty posting."
        )
        st.stop()

    # Matches training schema exactly: 'combined_text' + META_COLS, all metadata as strings
    row = pd.DataFrame([{
        "combined_text": combined,
        "location": "missing",
        "department": "missing",
        "salary_range": "missing",
        "telecommuting": str(int(telecommuting)),
        "has_company_logo": str(int(has_company_logo)),
        "has_questions": str(int(has_questions)),
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": "missing",
        "function": "missing",
    }])

    model = load_model()
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]

    if pred == 1:
        st.error(f"🚩 **Likely FRAUDULENT** (fraud probability: {proba:.1%})")
        st.write("Common red flags: vague company info, no logo, unrealistic pay, requests for personal/financial details.")
    else:
        st.success(f"✅ **Likely legitimate** (fraud probability: {proba:.1%})")
        st.write("Model prediction only. Always verify the employer independently before sharing personal information.")