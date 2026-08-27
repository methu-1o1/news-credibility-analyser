import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# Load test data
print("Loading test data...")

test_df = pd.read_csv("data/processed/test.csv")

print(f"Test rows: {len(test_df):,}")

# Separate text and labels
X_test_text = test_df["text"].fillna("")
y_test = test_df["label"]

# Basic preprocessing
print("\nPreprocessing test text...")

X_test_text = X_test_text.str.lower().str.strip()

print("Text preprocessing complete.")

# Load saved TF-IDF vectorizer
print("\nLoading saved TF-IDF vectorizer...")

tfidf = joblib.load("models/tfidf_vectorizer.joblib")

print("TF-IDF vectorizer loaded.")

# Transform test text
X_test = tfidf.transform(X_test_text)

print("Test text transformed.")

# Load saved Logistic Regression model
print("\nLoading saved Logistic Regression model...")

model = joblib.load("models/logistic_regression_model.joblib")

print("Logistic Regression model loaded.")

# Make predictions
print("\nMaking predictions...")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Predictions complete.")

# Get probability of predicted class
confidence = probabilities.max(axis=1)

# Add predictions and confidence to the dataframe
test_df["prediction"] = predictions
test_df["confidence"] = confidence

# Identify incorrect predictions
errors = test_df[test_df["label"] != test_df["prediction"]].copy()

print("\n===== ERROR ANALYSIS =====")

print(f"Total test rows: {len(test_df):,}")
print(f"Incorrect predictions: {len(errors):,}")
print(f"Correct predictions: {len(test_df) - len(errors):,}")

# Confusion matrix
cm = confusion_matrix(test_df["label"], test_df["prediction"])

print("\nConfusion matrix:")
print(cm)

# False positives
false_positives = test_df[
    (test_df["label"] == "FAKE") &
    (test_df["prediction"] == "REAL")
].copy()

# False negatives
false_negatives = test_df[
    (test_df["label"] == "REAL") &
    (test_df["prediction"] == "FAKE")
].copy()

print("\nFalse positives (FAKE predicted as REAL):")
print(len(false_positives))

print("\nFalse negatives (REAL predicted as FAKE):")
print(len(false_negatives))

# Save all errors
errors.to_csv(
    "data/processed/model_errors.csv",
    index=False
)

print("\nAll model errors saved to:")
print("data/processed/model_errors.csv")

# Show most confident incorrect predictions
most_confident_errors = errors.sort_values(
    "confidence",
    ascending=False
).head(20)

print("\n===== MOST CONFIDENT INCORRECT PREDICTIONS =====")

for index, row in most_confident_errors.iterrows():
    print("\n------------------------------")
    print(f"True label: {row['label']}")
    print(f"Prediction: {row['prediction']}")
    print(f"Confidence: {row['confidence']:.2%}")
    print(f"Text: {row['text'][:500]}")     