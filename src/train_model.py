import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load training and testing data
print("Loading training and testing data...")

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")

# Separate text and labels
X_train_text = train_df["text"].fillna("")
y_train = train_df["label"]

X_test_text = test_df["text"].fillna("")
y_test = test_df["label"]

# Basic text preprocessing
print("\nPreprocessing text...")

X_train_text = X_train_text.str.lower().str.strip()
X_test_text = X_test_text.str.lower().str.strip()

print("Text preprocessing complete.")

# Create TF-IDF vectorizer
print("\nCreating TF-IDF vectorizer...")

tfidf = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# Learn vocabulary ONLY from training data
X_train = tfidf.fit_transform(X_train_text)

# Transform test data using the SAME vocabulary
X_test = tfidf.transform(X_test_text)

print("TF-IDF transformation complete.")

print(f"\nTraining matrix shape: {X_train.shape}")
print(f"Testing matrix shape: {X_test.shape}")
print(f"Number of TF-IDF features: {len(tfidf.get_feature_names_out()):,}")    
from sklearn.linear_model import LogisticRegression

# Train Logistic Regression model
print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("Logistic Regression training complete.") 
import os
import joblib

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(model, "models/logistic_regression_model.joblib")

# Save the TF-IDF vectorizer
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")

print("\nModel and vectorizer saved successfully.") 