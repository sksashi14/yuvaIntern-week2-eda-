"""
Week 2 Task: Exploratory Data Analysis and Visualization
Dataset: Wine Recognition dataset distributed with scikit-learn.
Run:
    python -m pip install pandas numpy matplotlib seaborn scikit-learn
    python week2_eda_analysis.py
Outputs: figures/ and eda_summary.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

OUTPUT = Path("week2_eda_outputs")
FIGURES = OUTPUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# 1. Load public dataset
wine = load_wine(as_frame=True)
df = wine.frame.copy()
features = list(wine.feature_names)
class_names = {i: name for i, name in enumerate(wine.target_names)}
df["class_name"] = df["target"].map(class_names)

# 2. Initial inspection and quality checks
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())
print("\nExact duplicate rows:", int(df.duplicated().sum()))
print("\nClass counts:\n", df["class_name"].value_counts().sort_index())
print("\nDescriptive statistics:\n", df[features].describe().T)

# 3. Class distribution
counts = df["class_name"].value_counts().reindex(wine.target_names)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(counts.index, counts.values)
ax.bar_label(bars, padding=3)
ax.set(title="Observation Count by Wine Class", xlabel="Wine class", ylabel="Number of observations")
ax.set_ylim(0, max(counts.values) * 1.18)
fig.tight_layout()
fig.savefig(FIGURES / "01_class_distribution.png", bbox_inches="tight")
plt.close(fig)

# 4. Alcohol distribution by class
fig, ax = plt.subplots(figsize=(8, 5))
for name in wine.target_names:
    ax.hist(df.loc[df.class_name == name, "alcohol"], bins=10, alpha=0.45, label=name)
ax.set(title="Alcohol Distribution by Wine Class", xlabel="Alcohol", ylabel="Frequency")
ax.legend(title="Class")
fig.tight_layout()
fig.savefig(FIGURES / "02_alcohol_histogram.png", bbox_inches="tight")
plt.close(fig)

# 5. Boxplots for selected features
selected = ["alcohol", "malic_acid", "color_intensity", "proline"]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, feature in zip(axes.flat, selected):
    groups = [df.loc[df.class_name == name, feature].values for name in wine.target_names]
    ax.boxplot(groups, tick_labels=wine.target_names, showmeans=True)
    ax.set(title=f"{feature.replace('_', ' ').title()} by Class",
           xlabel="Wine class", ylabel=feature.replace("_", " ").title())
fig.tight_layout()
fig.savefig(FIGURES / "03_boxplots.png", bbox_inches="tight")
plt.close(fig)

# 6. Correlation heatmap
corr = df[features].corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
ax.set_title("Pearson Correlation Matrix of Chemical Features")
fig.tight_layout()
fig.savefig(FIGURES / "04_correlation_heatmap.png", bbox_inches="tight")
plt.close(fig)

# 7. Scatter plot
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.scatterplot(data=df, x="alcohol", y="color_intensity", hue="class_name",
                style="class_name", s=65, ax=ax)
ax.set_title("Alcohol vs. Colour Intensity by Wine Class")
ax.set_xlabel("Alcohol")
ax.set_ylabel("Color intensity")
fig.tight_layout()
fig.savefig(FIGURES / "05_alcohol_color_scatter.png", bbox_inches="tight")
plt.close(fig)

# 8. Class profile heatmap for features with the largest difference between class means
class_means = df.groupby("class_name")[features].mean()
separation = (class_means.max() - class_means.min()).sort_values(ascending=False)
top_features = list(separation.head(8).index)
means = class_means[top_features]
overall_mean = df[top_features].mean()
overall_std = df[top_features].std(ddof=0)
standardized = (means - overall_mean) / overall_std
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(standardized, annot=True, fmt=".1f", cmap="coolwarm", center=0, ax=ax,
            cbar_kws={"label": "Standard deviations from overall mean"})
ax.set_title("Class-Level Feature Profiles (Standardized Mean)")
ax.set_xlabel("Feature")
ax.set_ylabel("Wine class")
fig.tight_layout()
fig.savefig(FIGURES / "06_class_profile_heatmap.png", bbox_inches="tight")
plt.close(fig)

# 9. Proline histogram
fig, ax = plt.subplots(figsize=(8, 5))
for name in wine.target_names:
    ax.hist(df.loc[df.class_name == name, "proline"], bins=12, alpha=0.42, label=name)
ax.set(title="Proline Distribution by Wine Class", xlabel="Proline measurement", ylabel="Frequency")
ax.legend(title="Class")
fig.tight_layout()
fig.savefig(FIGURES / "07_proline_histogram.png", bbox_inches="tight")
plt.close(fig)

# 10. Correlations and IQR potential outliers
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
top_corr = upper.reindex(upper.abs().sort_values(ascending=False).index).head(10)
outlier_summary = []
for col in features:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((df[col] < lower) | (df[col] > upper_bound)).sum())
    outlier_summary.append((col, lower, upper_bound, count))
outlier_df = pd.DataFrame(outlier_summary, columns=["feature", "lower_bound", "upper_bound", "potential_outlier_count"])
outlier_df = outlier_df.sort_values("potential_outlier_count", ascending=False)

# 11. Save a text summary
summary_file = OUTPUT / "eda_summary.txt"
with summary_file.open("w", encoding="utf-8") as f:
    f.write("WINE RECOGNITION DATASET — EDA SUMMARY\n")
    f.write(f"Rows: {df.shape[0]}\nColumns including target and class label: {df.shape[1]}\n")
    f.write(f"Missing feature values: {int(df[features].isna().sum().sum())}\n")
    f.write(f"Exact duplicate rows: {int(df.duplicated().sum())}\n\n")
    f.write("Class counts:\n" + df["class_name"].value_counts().sort_index().to_string() + "\n\n")
    f.write("Descriptive statistics:\n" + df[features].describe().T.round(3).to_string() + "\n\n")
    f.write("Top absolute Pearson correlations:\n" + top_corr.round(3).to_string() + "\n\n")
    f.write("Potential outliers under the IQR rule:\n" + outlier_df.to_string(index=False) + "\n\n")
    f.write("Class-wise means:\n" + class_means.round(3).to_string() + "\n")

print(f"\nDone. Figures and summary saved under: {OUTPUT.resolve()}")
print(f"Top correlations:\n{top_corr.round(3)}")
print(f"Potential outlier flags:\n{outlier_df.head(8).to_string(index=False)}")
