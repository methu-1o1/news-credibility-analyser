import pandas as pd
import joblib

# Load test data
print("Loading test data...")

test_df = pd.read_csv("data/processed/test.csv")

print(f"Test rows: {len(test_df):,}")

# Prepare text
X_test_text = test_df["text"].fillna("")
X_test_text = X_test_text.str.lower().str.strip()

y_test = test_df["label"]

# Load TF-IDF vectorizer
print("\nLoading TF-IDF vectorizer...")

tfidf = joblib.load("models/tfidf_vectorizer.joblib")

print("TF-IDF vectorizer loaded.")

# Transform test data
X_test = tfidf.transform(X_test_text)

print("Test data transformed.")

# Load model
print("\nLoading Logistic Regression model...")

model = joblib.load("models/logistic_regression_model.joblib")

print("Logistic Regression model loaded.")

# Make predictions
print("\nMaking predictions...")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Predictions complete.")

# Calculate confidence
confidence = probabilities.max(axis=1)

# Create analysis dataframe
analysis_df = pd.DataFrame({
    "true_label": y_test.values,
    "prediction": predictions,
    "confidence": confidence
})

# Determine whether each prediction was correct
analysis_df["correct"] = (
    analysis_df["true_label"] == analysis_df["prediction"]
)

# Create confidence bins
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = [
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%"
]

analysis_df["confidence_range"] = pd.cut(
    analysis_df["confidence"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Calculate results for each confidence range
summary = analysis_df.groupby(
    "confidence_range",
    observed=False
).agg(
    total=("correct", "count"),
    correct=("correct", "sum")
)

summary["incorrect"] = summary["total"] - summary["correct"]

summary["accuracy"] = (
    summary["correct"] / summary["total"]
)

print("\n===== CONFIDENCE ANALYSIS =====")

print(summary)

# Save results
summary.to_csv(
    "data/processed/confidence_analysis.csv"
)

print("\nConfidence analysis saved to:")
print("data/processed/confidence_analysis.csv")    