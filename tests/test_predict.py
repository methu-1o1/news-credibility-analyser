import subprocess
import sys

import joblib  


MODEL_PATH = "models/logistic_regression_model.joblib"
TFIDF_PATH = "models/tfidf_vectorizer.joblib"


def load_model_and_vectorizer():
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return model, tfidf


def validate_input(text):
    text = text.strip()

    if not text:
        return False

    if len(text) < 20:
        return False

    letter_count = sum(character.isalpha() for character in text)

    if letter_count < 10:
        return False

    return True


def test_empty_input_is_rejected():
    assert validate_input("") is False


def test_short_input_is_rejected():
    assert validate_input("hello") is False


def test_symbol_only_input_is_rejected():
    assert validate_input("!!!!!!!!!!!!!!!!!!!!!!!!!") is False


def test_valid_headline_is_accepted():
    assert validate_input("Earthquake hits Japan") is True


def test_model_and_vectorizer_load():
    model, tfidf = load_model_and_vectorizer()

    assert model is not None
    assert tfidf is not None


def test_model_can_predict_valid_headline():
    model, tfidf = load_model_and_vectorizer()

    text = "Earthquake hits Japan"
    X = tfidf.transform([text.lower().strip()])

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    assert prediction in ["REAL", "FAKE"]
    assert len(probabilities) == 2
    assert abs(sum(probabilities) - 1.0) < 0.0001   

def run_predict(text):
    result = subprocess.run(
    [sys.executable, "-m", "src.predict"],
    input=text + "\n",
    text=True,
    capture_output=True,
    cwd="."
) 

    return result


def test_predict_rejects_short_input():
    result = run_predict("hello")

    assert "Input is too short to reliably analyze." in result.stdout


def test_predict_rejects_symbol_only_input():
    result = run_predict("!!!!!!!!!!!!!!!!!!!!!!!!!")

    assert "Input does not contain enough meaningful text to analyze." in result.stdout


def test_predict_accepts_valid_headline():
    result = run_predict("Earthquake hits Japan")

    assert "===== PREDICTION =====" in result.stdout
    assert "Prediction:" in result.stdout
    assert "Confidence:" in result.stdout
    assert "Vocabulary coverage:" in result.stdout


def test_predict_outputs_probabilities():
    result = run_predict("Earthquake hits Japan")

    assert "FAKE:" in result.stdout
    assert "REAL:" in result.stdout  