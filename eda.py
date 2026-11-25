import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("output", exist_ok=True)

df = pd.read_csv("data/titanic.csv")

print("=" * 50)
print("Titanic Dataset Loaded Successfully")
print("=" * 50)

print("\nFirst Five Rows")
print(df.head())

print("\nLast Five Rows")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nDataset Information")
df.info()

summary = df.describe(include="all")

print("\nStatistical Summary")
print(summary)

with open("output/statistics_summary.txt", "w") as f:
    f.write(str(summary))

print("\nMissing Values")

missing = df.isnull().sum()

print(missing)

print("\nNumPy Operations")

print("Average Age : ", np.nanmean(df["Age"]))

print("Maximum Fare : ", np.nanmax(df["Fare"]))

print("Minimum Fare : ", np.nanmin(df["Fare"]))

correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix")

print(correlation)

plt.figure(figsize=(10,8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("output/correlation_heatmap.png")

plt.show()

print("\nEDA Completed Successfully.")