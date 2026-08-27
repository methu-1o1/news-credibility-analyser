import pandas as pd
from pathlib import Path

# Paths
INPUT_PATH = Path("data/raw/RealFakeNews.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_PATH = OUTPUT_DIR / "cleaned_news.csv"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

original_rows = len(df)

print(f"Original rows: {original_rows:,}")

# --------------------------------------------------
# 1. Remove rows with missing text or labels
# --------------------------------------------------

df = df.dropna(subset=["text", "label"])

print(f"After removing missing values: {len(df):,}")

# --------------------------------------------------
# 2. Clean whitespace
# --------------------------------------------------

df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.upper()

# --------------------------------------------------
# 3. Remove very short texts
# --------------------------------------------------

before = len(df)

df = df[df["text"].str.len() >= 20]

print(f"Removed very short texts: {before - len(df):,}")

# --------------------------------------------------
# 4. Find texts with conflicting labels
# --------------------------------------------------

label_counts = df.groupby("text")["label"].nunique()

conflicting_texts = label_counts[label_counts > 1].index

before = len(df)

df = df[~df["text"].isin(conflicting_texts)]

print(f"Removed conflicting-label rows: {before - len(df):,}")

# --------------------------------------------------
# 5. Remove duplicate texts
# --------------------------------------------------

before = len(df)

df = df.drop_duplicates(subset=["text"])

print(f"Removed duplicate texts: {before - len(df):,}")

# --------------------------------------------------
# 6. Keep only valid labels
# --------------------------------------------------

before = len(df)

df = df[df["label"].isin(["REAL", "FAKE"])]

print(f"Removed invalid labels: {before - len(df):,}")

# --------------------------------------------------
# 7. Save cleaned dataset
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print()
print("Cleaning complete!")
print(f"Final rows: {len(df):,}")
print(f"Rows removed: {original_rows - len(df):,}")
print(f"Saved to: {OUTPUT_PATH}")

print()
print("Final label distribution:")
print(df["label"].value_counts())

print()
print("Final label proportions:")
print(df["label"].value_counts(normalize=True))                