"""Build the reproducible training-only Isolation Forest removal notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.strip().splitlines(keepends=True)}


cells = [
    markdown("""
# Isolation Forest removal and classification

Use the same 30,000 training rows and five folds as notebooks 03/04.
Isolation Forest is fitted on the 33-feature preprocessed training part of each fold.
Only training rows are removed. MI and classifier preprocessing are refitted
on retained rows. Every validation row is evaluated at threshold 0.5.
Classifier settings are fixed to the previous selected settings. No test rows are used.
"""),
    markdown("## 1. Environment, shared folds, and fixed models"),
    code(r'''
from pathlib import Path
import sys
import time
import platform
import json

experiment_started = time.perf_counter()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from IPython.display import display
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_info, threadpool_limits
get_ipython().run_line_magic("matplotlib", "inline")
thread_limit = threadpool_limits(limits=1)
cwd = Path.cwd()
repo_root = cwd if (cwd / "src/preprocessing.py").exists() else cwd.parent
assert (repo_root / "src/unsupervised_common.py").exists(), "Open from the repository root or notebooks/."
sys.path.insert(0, str(repo_root / "src"))
from unsupervised_common import (
    MODEL_SPECS, METRICS, WARNING_COLUMNS, load_training_data, run_outlier_cv,
    summarize_outlier_results, paired_outlier_differences,
    check_no_removal_baseline, pca_projection,
)
results_dir = repo_root / "results"
results_dir.mkdir(exist_ok=True)
print(f"Python {platform.python_version()}; sklearn {sklearn.__version__}; pandas {pd.__version__}; numpy {np.__version__}")
data = load_training_data(repo_root)
print(f"CV rows: {len(data.X_dev):,}; original features: {data.X_dev.shape[1]}; folds: {len(data.folds)}")
display(pd.DataFrame([{"model": name, "k": spec["k"], "params": json.dumps(spec["params"])}
                      for name, spec in MODEL_SPECS.items()]))
'''),
    markdown("""
## 2. Training-only removal and five-fold evaluation

Use Isolation Forest with 100 trees, random_state=42, and n_jobs=1. Fit each
detector once per fold and share its mask across the four classifiers. Detector
preprocessing and fit_predict are timed separately. Classifier timing matches
notebooks 03/04. Captured fit diagnostics are saved without changing settings.
"""),
    code(r'''
def make_iforest(contamination):
    return IsolationForest(n_estimators=100, contamination=contamination,
                           random_state=42, n_jobs=1)

CONDITIONS = [("No removal", None), ("Isolation Forest 1%", 0.01), ("Isolation Forest 5%", 0.05)]
runs = {}
for condition, contamination in CONDITIONS:
    print(f"Running {condition}", flush=True)
    runs[condition] = run_outlier_cv(data, make_iforest, contamination, condition)
    fold_results = pd.concat([run["folds"] for run in runs.values()], ignore_index=True)
    removals = pd.concat([run["removals"] for run in runs.values()], ignore_index=True)
    diagnostic_frames = [run["warnings"] for run in runs.values() if not run["warnings"].empty]
    diagnostics = (pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames
                   else pd.DataFrame(columns=WARNING_COLUMNS))
    fold_results.to_csv(results_dir / "iforest_cv_folds.csv", index=False)
    removals.to_csv(results_dir / "iforest_removals.csv", index=False)
    diagnostics.to_csv(results_dir / "iforest_fit_diagnostics.csv", index=False)
money = pd.concat([run["money"] for run in runs.values() if not run["money"].empty], ignore_index=True)
members = pd.concat([run["members"] for run in runs.values() if not run["members"].empty], ignore_index=True)
money.to_csv(results_dir / "iforest_money_statistics.csv", index=False)
members.to_csv(results_dir / "iforest_removed_rows.csv", index=False)
lof_folds = pd.read_csv(results_dir / "lof_cv_folds.csv")
iforest_folds = fold_results.loc[fold_results.condition != "No removal"]
combined_cv_folds = pd.concat([lof_folds, iforest_folds], ignore_index=True)
combined_cv_folds.to_csv(results_dir / "outlier_cv_folds.csv", index=False)
print(f"Captured fit diagnostics: {len(diagnostics)}")
if not diagnostics.empty:
    display(diagnostics.groupby(["condition", "phase", "category"]).size().rename("count").reset_index())
'''),
    markdown("""
## 3. No-removal reference checks

LR and k-NN AP are checked fold by fold against the existing best-setting CSV.
RF and GaussianNB mean AP are checked against notebook 03's four-decimal values.
The respective absolute tolerances are 1e-12 and 5e-5.
"""),
    code(r'''
baseline_checks = check_no_removal_baseline(fold_results, repo_root)
baseline_checks.to_csv(results_dir / "iforest_baseline_checks.csv", index=False)
display(baseline_checks)
'''),
    markdown("""
## 4. Metrics and paired fold differences

SD is the sample standard deviation across five folds. Each difference is
removal minus no removal for the same model and fold.
"""),
    code(r'''
summary = summarize_outlier_results(fold_results)
paired_folds, paired_summary = paired_outlier_differences(fold_results)
summary.to_csv(results_dir / "iforest_cv_summary.csv", index=False)
paired_folds.to_csv(results_dir / "iforest_paired_folds.csv", index=False)
paired_summary.to_csv(results_dir / "iforest_paired_summary.csv", index=False)
metric_columns = [column for metric in METRICS for column in [f"{metric}_mean", f"{metric}_sd"]]
display(summary[["condition", "model"] + metric_columns].round(6))
display(paired_summary.round(6))
ap_table = summary[["condition", "model", "AP_mean", "AP_sd"]].merge(
    paired_summary[["condition", "model", "AP_difference_mean", "AP_difference_sd"]],
    on=["condition", "model"], validate="one_to_one"
)
display(ap_table.round(6))
combined_summary = summarize_outlier_results(combined_cv_folds)
combined_paired_folds, combined_paired_summary = paired_outlier_differences(combined_cv_folds)
combined_summary.to_csv(results_dir / "outlier_cv_summary.csv", index=False)
combined_paired_folds.to_csv(results_dir / "outlier_paired_folds.csv", index=False)
combined_paired_summary.to_csv(results_dir / "outlier_paired_summary.csv", index=False)
combined_ap_table = combined_summary[["condition", "model", "AP_mean", "AP_sd"]].merge(
    combined_paired_summary[["condition", "model", "AP_difference_mean", "AP_difference_sd"]],
    on=["condition", "model"], validate="one_to_one"
)
display(combined_ap_table.round(6))
fig, ax = plt.subplots(figsize=(10, 4.5))
plot_conditions = ["No removal", "LOF 1%", "LOF 5%", "Isolation Forest 1%", "Isolation Forest 5%"]
for name in MODEL_SPECS:
    group = combined_summary.loc[combined_summary.model == name].set_index("condition").loc[plot_conditions]
    ax.errorbar(range(len(plot_conditions)), group.AP_mean, yerr=group.AP_sd,
                marker="o", capsize=3, label=name)
ax.set_xticks(range(len(plot_conditions)), plot_conditions, rotation=15)
ax.set(xlabel="training-row condition", ylabel="AP mean ± SD", title="Five-fold validation AP")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
'''),
    markdown("## 5. Actual removal rates and raw monetary summaries"),
    code(r'''
display(removals[["condition", "fold", "contamination", "training_count", "removed_count",
                  "removed_fraction", "training_high_income_rate", "removed_high_income_rate",
                  "retained_high_income_rate"]].round(6))
with pd.option_context("display.max_rows", None):
    display(money.round(4))
'''),
    markdown("""
## 6. Fold 1 visualization

PCA is fitted to the full preprocessed fold-1 training matrix for visualization
only. Both conditions use the same projection. Validation rows are not plotted.
"""),
    code(r'''
first_one = runs["Isolation Forest 1%"]["first_fold"]
first_five = runs["Isolation Forest 5%"]["first_fold"]
assert np.array_equal(first_one["row_index"], first_five["row_index"])
assert (first_one["Z"] != first_five["Z"]).nnz == 0
coordinates, pca = pca_projection(first_one["Z"])
pd.DataFrame({"component": [1, 2], "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(
    results_dir / "iforest_pca_variance.csv", index=False
)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, condition in zip(axes, ["Isolation Forest 1%", "Isolation Forest 5%"]):
    mask = runs[condition]["first_fold"]["removed"]
    ax.scatter(coordinates[~mask, 0], coordinates[~mask, 1], s=5, alpha=0.35,
               color="steelblue", label="retained")
    ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=18, alpha=0.75,
               marker="x", color="crimson", label="removed")
    ax.set(title=f"{condition}: fold 1 training; PCA only", xlabel="PC1", ylabel="PC2")
    ax.legend()
fig.tight_layout()
plt.show()
'''),
    markdown("## 7. Final checks and execution time"),
    code(r'''
assert len(data.X_train) == 172_459 and len(data.X_dev) == 30_000
assert data.X_dev.shape[1] == 33 and len(data.folds) == 5
assert len(fold_results) == 60
assert fold_results.groupby(["condition", "model"]).size().eq(5).all()
assert len(summary) == 12 and len(paired_summary) == 12 and len(paired_folds) == 60
assert len(removals) == 15 and len(money) == 80
assert len(combined_cv_folds) == 100
assert combined_cv_folds.groupby(["condition", "model"]).size().eq(5).all()
assert set(combined_cv_folds.condition) == {
    "No removal", "LOF 1%", "LOF 5%", "Isolation Forest 1%", "Isolation Forest 5%"
}
assert len(combined_paired_folds) == 100
assert fold_results.validation_count.eq(6_000).all()
assert fold_results[["TN", "FP", "FN", "TP"]].sum(axis=1).eq(6_000).all()
assert removals.loc[removals.condition == "No removal", "removed_count"].eq(0).all()
assert np.allclose(removals.removed_fraction, removals.removed_count / 24_000)
for condition, contamination in [("Isolation Forest 1%", 0.01), ("Isolation Forest 5%", 0.05)]:
    observed = removals.loc[removals.condition == condition, "removed_fraction"]
    assert np.allclose(observed, contamination, atol=1 / 24_000, rtol=0)
assert members.groupby(["condition", "fold"]).size().sum() == removals.removed_count.sum()
assert baseline_checks.matches.all()
assert fold_results.loc[fold_results.k == 33, "MI_seconds"].eq(0).all()
assert all(item["num_threads"] == 1 for item in threadpool_info())
total_seconds = time.perf_counter() - experiment_started
pd.DataFrame([{"total_seconds": total_seconds, "captured_fit_diagnostics": len(diagnostics),
               "python": platform.python_version(), "sklearn": sklearn.__version__,
               "pandas": pd.__version__, "numpy": np.__version__, "native_threads": 1}]).to_csv(
    results_dir / "iforest_execution.csv", index=False
)
print("Final checks passed.")
print(f"Total execution time: {total_seconds:.2f} seconds")
'''),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {
    "display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells, start=1):
    cell["id"] = f"iforest-{index:02d}"
path = Path(__file__).resolve().parents[1] / "notebooks/08_isolation_forest_outliers.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
print(f"Built {path}")
