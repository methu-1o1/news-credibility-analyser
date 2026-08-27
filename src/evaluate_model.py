import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
) 

# Load the test dataset
print("Loading test data...")

test_df = pd.read_csv("data/processed/test.csv")

print(f"Test rows: {len(test_df):,}")

# Separate text and labels
X_test_text = test_df["text"].fillna("")
y_test = test_df["label"]

# Apply the same basic preprocessing used during training
print("\nPreprocessing test text...")

X_test_text = X_test_text.str.lower().str.strip()

print("Text preprocessing complete.")

# Load the saved TF-IDF vectorizer
print("\nLoading saved TF-IDF vectorizer...")

tfidf = joblib.load("models/tfidf_vectorizer.joblib")

# Transform test text using the saved vectorizer
X_test = tfidf.transform(X_test_text)

print("Test text transformed using saved TF-IDF vectorizer.")

# Load the saved Logistic Regression model
print("\nLoading saved Logistic Regression model...")

model = joblib.load("models/logistic_regression_model.joblib")

print("Model loaded successfully.")

# Make predictions
print("\nMaking predictions...")

y_pred = model.predict(X_test)

print("Predictions complete.")

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="FAKE")
recall = recall_score(y_test, y_pred, pos_label="FAKE")
f1 = f1_score(y_test, y_pred, pos_label="FAKE")

# Display results
print("\n===== MODEL EVALUATION =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")                                        
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["REAL", "FAKE"])

print("\n===== CONFUSION MATRIX =====")
print(cm)

# Create and save confusion matrix visualization
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["REAL", "FAKE"]
)

display.plot()

import matplotlib.pyplot as plt

plt.title("News Credibility Classifier - Confusion Matrix")
plt.savefig("models/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.savefig("models/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nConfusion matrix saved to models/confusion_matrix.png")   