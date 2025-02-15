import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define class mapping (excluding background "Class_0")
class_mapping = {
    "Class_1": "blackheads",
    "Class_2": "dark spot",
    "Class_3": "nodules",
    "Class_4": "papules",
    "Class_5": "pustules",
    "Class_6": "whiteheads"
}

# Load CSV
file_path = "./conf_matrix.csv"
df = pd.read_csv(file_path)

# Remove "Class_0" (background class) rows and columns
df = df[~df["Actual"].eq("Class_0")]  # Remove rows where Actual == "Class_0"
df = df[~df["Predicted"].eq("Class_0")]  # Remove rows where Predicted == "Class_0"

# Map class labels
df["Actual"] = df["Actual"].map(class_mapping)
df["Predicted"] = df["Predicted"].map(class_mapping)

# Pivot into a square confusion matrix
conf_matrix = df.pivot(index="Actual", columns="Predicted", values="nPredictions").fillna(0)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt=".0f", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# Save the plot
plt.savefig("confusion_matrix.png")
plt.close()

print("Confusion matrix plot saved as 'confusion_matrix.png'")
