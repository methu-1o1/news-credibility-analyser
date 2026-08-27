import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
INPUT_PATH = Path("data/processed/cleaned_news.csv")
OUTPUT_DIR = Path("data/processed")

TRAIN_PATH = OUTPUT_DIR / "train.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"

# Load cleaned dataset
print("Loading cleaned dataset...")
df = pd.read_csv(INPUT_PATH)

print(f"Total rows: {len(df):,}")

# Split into training and testing sets
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Save the datasets
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

# Display results
print()
print("Split complete!")
print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")

print()
print("Training label distribution:")
print(train_df["label"].value_counts())
print(train_df["label"].value_counts(normalize=True))

print()
print("Testing label distribution:")
print(test_df["label"].value_counts())
print(test_df["label"].value_counts(normalize=True))

print()
print(f"Training data saved to: {TRAIN_PATH}")
print(f"Testing data saved to: {TEST_PATH}")   