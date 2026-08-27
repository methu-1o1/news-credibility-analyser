import pandas as pd
import matplotlib.pyplot as plt

# Load confidence analysis results
print("Loading confidence analysis...")

df = pd.read_csv("data/processed/confidence_analysis.csv")

print("Confidence analysis loaded.")
print(df)

# Convert accuracy to percentage
df["accuracy_percent"] = df["accuracy"] * 100

# Create the plot
plt.figure(figsize=(10, 6))

plt.bar(
    df["confidence_range"],
    df["accuracy_percent"]
)

plt.xlabel("Model Confidence Range")
plt.ylabel("Actual Accuracy (%)")
plt.title("Model Confidence vs Actual Accuracy")

plt.ylim(0, 100)

# Add accuracy values above each bar
for index, row in df.iterrows():
    plt.text(
        index,
        row["accuracy_percent"] + 1,
        f"{row['accuracy_percent']:.1f}%",
        ha="center"
    )

plt.tight_layout()

# Save the graph
output_path = "data/processed/confidence_accuracy.png"

plt.savefig(output_path, dpi=300)

print(f"\nConfidence graph saved to: {output_path}")

plt.show()   