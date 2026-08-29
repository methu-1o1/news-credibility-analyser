import pandas as pd

# Load original dataset
original = pd.read_csv("data/raw/RealFakeNews.csv")

# --- Onion or Not: satire examples (label == 1) ---
onion_or_not = pd.read_csv("data/raw/onion_or_not.csv")
onion_satire_rows = onion_or_not[onion_or_not["label"] == 1].copy()
onion_satire_rows = onion_satire_rows.rename(columns={"text": "text"})
onion_satire_rows["label"] = "FAKE"
onion_satire_rows = onion_satire_rows[["text", "label"]]
onion_satire_rows = onion_satire_rows.sample(n=3000, random_state=42)

# --- Clickbait dataset: clickbait examples (clickbait == 1) ---
clickbait = pd.read_csv("data/raw/clickbait_dataset.csv")
clickbait_rows = clickbait[clickbait["clickbait"] == 1].copy()
clickbait_rows = clickbait_rows.rename(columns={"headline": "text"})
clickbait_rows["label"] = "FAKE"
clickbait_rows = clickbait_rows[["text", "label"]]
clickbait_rows = clickbait_rows.sample(n=3000, random_state=42)

# --- Onion satirical: pure satire, all rows are satire ---
onion_satirical = pd.read_csv("data/raw/onion_satirical.csv")
onion_satirical_rows = onion_satirical.rename(columns={"Title": "text"})
onion_satirical_rows["label"] = "FAKE"
onion_satirical_rows = onion_satirical_rows[["text", "label"]]
onion_satirical_rows = onion_satirical_rows.sample(n=3000, random_state=42)

# --- Combine everything ---
combined = pd.concat(
    [original, onion_satire_rows, clickbait_rows, onion_satirical_rows],
    ignore_index=True
)

print("Original size:", original.shape)
print("Added Onion-or-Not satire rows:", onion_satire_rows.shape)
print("Added clickbait rows:", clickbait_rows.shape)
print("Added Onion satirical rows:", onion_satirical_rows.shape)
print("Combined size:", combined.shape)
print("\nNew label balance:")
print(combined["label"].value_counts())

combined.to_csv("data/raw/RealFakeNews_augmented.csv", index=False)
print("\nSaved to data/raw/RealFakeNews_augmented.csv")   