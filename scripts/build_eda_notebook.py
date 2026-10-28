"""Build the reproducible EDA notebook. Run from the repository root."""

from pathlib import Path

import nbformat as nbf


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


notebook = nbf.v4.new_notebook()
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
notebook.metadata["language_info"] = {"name": "python"}
notebook.cells = [
    markdown("""
# Census-Income (KDD): EDA

**Conclusion: Compare 33 features after removing columns with year-specific recording and constant columns.** This notebook examines missingness, distributions, and associations in 215,574 adults from the two UCI files.

Environment: Colab or Kaggle. The first code cell downloads the data if it is not available.
"""),
    code("""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

cwd = Path.cwd()
repo_root = cwd if (cwd / "src" / "data_loader.py").exists() else cwd.parent
assert (repo_root / "src" / "data_loader.py").exists(), "Run this notebook from the repository root or the notebooks/ directory."
data_path = repo_root / "data" / "processed" / "census_income_adults.csv"
if not data_path.exists():
    subprocess.run([sys.executable, "src/data_loader.py"], cwd=repo_root, check=True)

df = pd.read_csv(data_path, skipinitialspace=True)
assert df.shape == (215_574, 41)
assert "instance_weight" not in df.columns
print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
display(df.head(3))
"""),
    markdown("""
## 1. Check missing values

**Rule: Treat only `?` as missing.** Keep `Not in universe` as a category indicating that the field does not apply.
"""),
    code("""
# Strip whitespace from string columns and treat only ? as missing.
text_cols = df.select_dtypes(include=["object", "string"]).columns
df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())
df[text_cols] = df[text_cols].replace("?", pd.NA)

missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_rate_pct": df.isna().mean().mul(100).round(2),
}).sort_values("missing_count", ascending=False)
display(missing[missing.missing_count > 0])
"""),
    markdown("## Data quality: exact duplicates"),
    code("""
duplicate_count = int(df.duplicated().sum())
duplicate_group_rows = int(df.duplicated(keep=False).sum())
predictor_columns = [col for col in df.columns if col != "income_class"]
conflicting_predictors = int((
    df.groupby(predictor_columns, dropna=False)["income_class"].nunique() > 1
).sum())
print(f"Additional exact duplicate rows: {duplicate_count:,} ({duplicate_count / len(df):.2%})")
print(f"Rows belonging to exact-duplicate groups: {duplicate_group_rows:,}")
print(f"Predictor patterns with conflicting income labels: {conflicting_predictors:,}")
"""),
    markdown("""There are 16,078 duplicate lines (7.46%) that completely match other lines, excluding the first occurrence. However, there is no ID to identify individuals, so we cannot claim that these lines are duplicate records of the same person. Therefore, we did not eliminate the data for this reason. At the analysis phase, we will continuously pay attention to the possibility that lines with the same values appear in both the training and assessment data."""),
    markdown("## 2. Proportion of `Not in universe` values (check constant columns)"),
    code("""
not_in_universe = pd.DataFrame({
    "count": df.eq("Not in universe").sum(),
    "rate_pct": df.eq("Not in universe").mean().mul(100).round(2),
}).sort_values("count", ascending=False)
display(not_in_universe[not_in_universe["count"] > 0])
"""),
    markdown("## 3. Convert five numeric codes to categorical type"),
    code("""
# These columns are nominal in .names but are read as integers from the CSV.
numeric_categorical = [
    "detailed_industry_recode", "detailed_occupation_recode",
    "own_business_or_self_employed", "veterans_benefits", "year",
]
print("Before:")
display(df[numeric_categorical].dtypes.to_frame("dtype"))
for col in numeric_categorical:
    df[col] = df[col].astype("string")
print("After:")
display(df[numeric_categorical].dtypes.to_frame("dtype"))

numeric_cols = [
    "age", "wage_per_hour", "capital_gains", "capital_losses",
    "dividends_from_stocks", "num_persons_worked_for_employer",
    "weeks_worked_in_year",
]
assert set(df.select_dtypes(include="number").columns) == set(numeric_cols)
display(df[numeric_cols].describe(percentiles=[.5, .95, .99]).T)
"""),
    markdown("## 4. High income is 8.61% of the data (class imbalance)"),
    code("""
class_counts = df["income_class"].value_counts(dropna=False)
display(pd.DataFrame({
    "count": class_counts,
    "rate_pct": class_counts.div(len(df)).mul(100).round(2),
}))
"""),
    markdown("""
## 5. Histograms of seven numeric features by income class

**Purpose: Compare distributions across income classes.** `log1p` is used to display monetary features; it does not change the data.
"""),
    code("""
skewed_cols = {"wage_per_hour", "capital_gains", "capital_losses", "dividends_from_stocks"}
classes = sorted(df["income_class"].dropna().unique())
colors = ["#2878B5", "#D66027"]
fig, axes = plt.subplots(4, 2, figsize=(14, 17))
for ax, col in zip(axes.flat, numeric_cols):
    for label, color in zip(classes, colors):
        values = df.loc[df["income_class"] == label, col].dropna().to_numpy()
        if col in skewed_cols:
            values = np.log1p(values)
        ax.hist(values, bins=40, density=True, histtype="step", linewidth=1.6,
                color=color, label=label)
    ax.set_title(col)
    ax.set_xlabel("log1p(value)" if col in skewed_cols else "value")
    ax.set_ylabel("density within class")
    ax.legend(fontsize=8)
axes.flat[-1].axis("off")
fig.suptitle("Numeric distributions by income class", fontsize=15)
fig.tight_layout()
plt.show()
"""),
    markdown("""
## 6. Box plots of seven numeric features by income class

**Purpose: Compare the center and spread of each distribution.** Outlier points are hidden. `log1p` is used to display monetary features.
"""),
    code("""
fig, axes = plt.subplots(4, 2, figsize=(14, 17))
for ax, col in zip(axes.flat, numeric_cols):
    groups = []
    for label in classes:
        values = df.loc[df["income_class"] == label, col].dropna().to_numpy()
        groups.append(np.log1p(values) if col in skewed_cols else values)
    ax.boxplot(groups, tick_labels=classes, showfliers=False)
    ax.set_title(col)
    ax.set_ylabel("log1p(value)" if col in skewed_cols else "value")
    ax.tick_params(axis="x", labelrotation=15)
axes.flat[-1].axis("off")
fig.suptitle("Numeric boxplots by income class (outlier points hidden)", fontsize=15)
fig.tight_layout()
plt.show()

"""),
    markdown("""
### Outlier summary

**Conclusion: Do not remove outliers from monetary columns wholesale.** Many values are zero, making the all-row IQR zero. Report outlier rates among nonzero values and high-income rates for zero versus nonzero values. It is unconfirmed whether 99,999 is a top-code.
"""),
    code("""
def iqr_outlier_rate(values):
    q1, q3 = values.quantile([.25, .75])
    iqr = q3 - q1
    if iqr == 0:
        return np.nan  # The IQR rule cannot be applied when the IQR is zero.
    return ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean() * 100

is_high = df["income_class"].eq("50000+.")
outlier_rows = []
for col in numeric_cols:
    values = df[col]
    nonzero = values > 0
    outlier_rows.append({
        "feature": col,
        "zero_rate_pct": (~nonzero).mean() * 100,
        "IQR_outlier_rate_all_pct": iqr_outlier_rate(values),
        "IQR_outlier_rate_nonzero_pct": iqr_outlier_rate(values[nonzero]),
        "max": values.max(),
        "rows_at_max": int(values.eq(values.max()).sum()),
        "high_income_rate_if_nonzero_pct": is_high[nonzero].mean() * 100,
        "high_income_rate_if_zero_pct": is_high[~nonzero].mean() * 100,
    })
display(pd.DataFrame(outlier_rows).set_index("feature").round(2))
"""),
    markdown("## 7. Correlations among seven numeric features (two are employment-related)"),
    code("""
corr = df[numeric_cols].corr(method="pearson")
fig, ax = plt.subplots(figsize=(9, 7))
heat = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(numeric_cols)), labels=numeric_cols, rotation=55, ha="right")
ax.set_yticks(range(len(numeric_cols)), labels=numeric_cols)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(heat, ax=ax, label="Pearson r")
fig.tight_layout()
plt.show()
"""),
    markdown("""
## 8. Associations among categorical features (Cramér's V)

**Conclusion: Do not drop a feature solely because it has a strong association with another.** Compute Cramér's V (0–1) between categorical features and the target. Count missing values as `Missing` and exclude constant columns.
"""),
    code("""
from scipy.stats.contingency import association

categorical_cols = [c for c in df.columns if c not in numeric_cols and c != "income_class"]
constant_cols = [c for c in categorical_cols if df[c].nunique(dropna=False) == 1]
print("Excluded constant columns:", constant_cols)
cat_frame = df[[c for c in categorical_cols if c not in constant_cols] + ["income_class"]]
codes = {c: pd.factorize(cat_frame[c].fillna("Missing"))[0] for c in cat_frame.columns}

def cramers_v(a, b):
    n_a, n_b = a.max() + 1, b.max() + 1
    table = np.bincount(a * n_b + b, minlength=n_a * n_b).reshape(n_a, n_b)
    return association(table, method="cramer")

names = list(codes)
cramer = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
for i, left in enumerate(names):
    for right in names[i + 1:]:
        cramer.loc[left, right] = cramer.loc[right, left] = cramers_v(codes[left], codes[right])

order = cramer["income_class"].sort_values(ascending=False).index
cramer = cramer.loc[order, order]
fig, ax = plt.subplots(figsize=(15, 13))
heat = ax.imshow(cramer, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(range(len(order)), labels=order, rotation=70, ha="right", fontsize=8)
ax.set_yticks(range(len(order)), labels=order, fontsize=8)
for i in range(len(order)):
    for j in range(len(order)):
        if i != j and cramer.iloc[i, j] >= 0.5:
            ax.text(j, i, f"{cramer.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white")
fig.colorbar(heat, ax=ax, label="Cramér's V")
ax.set_title("Association between categorical features and income class (V >= 0.5 labelled)")
fig.tight_layout()
plt.show()

pairs = cramer.drop(index="income_class", columns="income_class").stack()
pairs = pairs[[a < b for a, b in pairs.index]].sort_values(ascending=False)
print("Most associated feature pairs:")
display(pairs.head(15).round(3).to_frame("cramers_v"))
print("Association with income_class:")
display(cramer["income_class"].drop("income_class").round(3).to_frame("cramers_v"))
"""),
    markdown("""
## Notes and observations

**Conclusion: Use 33 features as the baseline and do not remove outliers wholesale.**

- **Excluded:** Four migration columns that are entirely missing in 1995; `family_members_under_18`, which is constant among adults; and `live_in_this_house_1_year_ago` and `full_or_part_time_employment_stat`, which are effectively constant in one year. Excluding the last two may discard signal from the year in which they vary.
- **Missingness:** Treat only `?` as missing. Keep `Not in universe` as a category. Missing rates for the three country-of-birth columns range from 2.09% to 3.88%.
- **Class balance:** There are 18,566 high-income records (8.61%). Use stratified splits and average precision (AP) as the primary metric. Choose class weighting with training cross-validation.
- **Numeric features:** The four monetary columns contain 85–97% zeros. `capital_gains` has 578 values of 99,999; it is unconfirmed whether this is a top-code. Compare raw and `log1p` values without removing outliers.
- **Associations:** The two employment-related features have a Pearson correlation of about 0.66. Detailed and broad occupation/industry features have Cramér's V = 1.00, but may not contain identical information, so keep them.
- **Next steps:** Fit imputation, scaling, one-hot encoding, and mutual information (MI) on training data only. Use PCA for visualization and do not use the final test set to make decisions.
"""),
]

path = Path("notebooks/01_eda.ipynb")
path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, path)
print(f"Built {path}")
