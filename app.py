import streamlit as st
import joblib

from src.credibility_features import analyse_text, interpret_features
from src.combined_assessment import create_assessment

# Load model and vectorizer once
tfidf = joblib.load("models/tfidf_vectorizer.joblib")
model = joblib.load("models/logistic_regression_model.joblib")

st.title("AI-Powered News Credibility Analyser")
st.write("Enter a news headline or article below to check its credibility.")

text = st.text_area("News headline or article:")
st.info(
    "This tool provides a machine-generated credibility signal, not a fact-check. "
    "Always verify important claims using trusted, independent sources."
) 
if st.button("Analyse"):
    text = text.strip()

    if len(text) < 20:
        st.error("Input is too short to reliably analyze. Please enter more information.")
    else:
        processed_text = text.lower()
        X = tfidf.transform([processed_text])

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = max(probabilities)

        features = analyse_text(text)
        indicators = interpret_features(features)

        assessment = create_assessment(prediction, confidence, indicators["linguistic_risk"])

        verdict = assessment["final_verdict"] 

        st.subheader("Overall Assessment")
        if verdict == "Likely Unreliable":
            st.error(f"## {verdict}")
        elif verdict == "Uncertain — Verify":
            st.warning(f"## {verdict}")
        else:
            st.success(f"## {verdict}")

        st.write(assessment["overall_assessment"])

        with st.expander("See detailed breakdown"):
            st.write(f"**ML model prediction:** {prediction} (confidence: {confidence:.2%})")
            st.write(f"**Sensational language:** {indicators['sensational_language']}")
            st.write(f"**Emotional language:** {indicators['emotional_language']}")
            st.write(f"**Linguistic risk:** {indicators['linguistic_risk']}")