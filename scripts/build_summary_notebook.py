"""Build a read-only analysis of saved results; no estimators are fitted."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.strip().splitlines(keepends=True)}


cells = [
    markdown("""
# Summary analysis of saved results

Read saved outputs from notebooks 03–08 and their CSVs. Do not refit models,
recompute MI, or evaluate test rows. Cluster descriptions reuse saved assignments
and the same training-only sample. Install `lxml` before running this notebook.
"""),
    markdown("## 1. Environment and inputs"),
    code(r'''
from pathlib import Path
from io import StringIO
from itertools import combinations
import ast
import hashlib
import json
import platform
import re
import sys
import time

experiment_started = time.perf_counter()
import lxml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from IPython.display import display
from threadpoolctl import threadpool_info, threadpool_limits
get_ipython().run_line_magic("matplotlib", "inline")
thread_limit = threadpool_limits(limits=1)
cwd = Path.cwd()
repo_root = cwd if (cwd / "src/unsupervised_common.py").exists() else cwd.parent
assert (repo_root / "src/unsupervised_common.py").exists(), "Open from the repository root or notebooks/."
sys.path.insert(0, str(repo_root / "src"))
from preprocessing import NUMERIC_COLUMNS
from unsupervised_common import load_training_data, clustering_sample
results_dir = repo_root / "results"
K_VALUES = [5, 10, 15, 17, 20, 25, 33]
MODELS = ["Random Forest", "GaussianNB", "Logistic Regression", "k-NN"]
TIME_COLUMNS = ["MI_seconds_mean", "preprocessing_seconds_mean", "fit_seconds_mean", "predict_seconds_mean"]
protected_paths = [p for folder in ["notebooks", "scripts", "src", "results"]
                   for p in (repo_root / folder).iterdir() if p.is_file()
                   and not p.name.startswith(("09_summary_analysis", "build_summary_notebook", "summary_"))]
input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths}
print(f"Python {platform.python_version()}; sklearn {sklearn.__version__}; pandas {pd.__version__}; numpy {np.__version__}")
'''),
    markdown("""
## 2. Initial-CV timing, including feature selection

RF/GaussianNB times come from notebook 03's saved HTML summary, rounded to four
decimal places. LR/k-NN times retain the CSV's saved precision. Totals sum the
four recorded mean times; ratios use each model's k=33 total as the denominator.
"""),
    code(r'''
path03 = repo_root / "notebooks/03_classification_comparison.ipynb"
saved03 = json.loads(path03.read_text())
candidates = []
for cell_index, cell in enumerate(saved03["cells"]):
    for output in cell.get("outputs", []):
        if "text/html" not in output.get("data", {}):
            continue
        html = "".join(output["data"]["text/html"])
        for frame in pd.read_html(StringIO(html), flavor="lxml", index_col=0):
            if {"model", "k", *TIME_COLUMNS}.issubset(frame.columns):
                candidates.append((cell_index, frame, html))
assert len(candidates) == 1, "Cannot uniquely read the saved initial-CV HTML summary; stop here."
source_cell, saved_times, source_html = candidates[0]
source_code = "".join(saved03["cells"][source_cell]["source"])
rounding = [node.args[0].value for node in ast.walk(ast.parse(source_code))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "round" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "summary" and node.args
            and isinstance(node.args[0], ast.Constant)]
saved_numeric_strings = re.findall(r"<td>\s*(-?\d+\.\d+)\s*</td>", source_html)
observed_decimal_places = sorted({len(value.split(".")[1]) for value in saved_numeric_strings})
assert rounding == [4] and observed_decimal_places == [4]
assert len(saved_times) == 14 and set(saved_times.model) == {"Random Forest", "GaussianNB"}
assert set(saved_times.k) == set(K_VALUES)
csv_times = pd.read_csv(results_dir / "lr_knn_cv_summary.csv")
initial_times = pd.concat([
    saved_times[["model", "k"] + TIME_COLUMNS].assign(source="03 saved HTML", source_decimal_places=4),
    csv_times[["model", "k"] + TIME_COLUMNS].assign(source="lr_knn_cv_summary.csv", source_decimal_places=np.nan),
], ignore_index=True)
initial_times["total_with_MI_seconds_mean"] = initial_times[TIME_COLUMNS].sum(axis=1)
initial_times["total_without_MI_seconds_mean"] = initial_times[TIME_COLUMNS[1:]].sum(axis=1)
k33_total = initial_times.loc[initial_times.k == 33].set_index("model")["total_with_MI_seconds_mean"]
initial_times["relative_to_k33"] = initial_times.total_with_MI_seconds_mean / initial_times.model.map(k33_total)
initial_times["relative_without_MI_to_k33"] = initial_times.total_without_MI_seconds_mean / initial_times.model.map(k33_total)
initial_times["_order"] = initial_times.model.map({name: i for i, name in enumerate(MODELS)})
initial_times = initial_times.sort_values(["_order", "k"]).drop(columns="_order").reset_index(drop=True)
initial_times.to_csv(results_dir / "summary_initial_cv_times.csv", index=False)
precision = pd.DataFrame([
    {"source": "03 saved HTML", "models": "Random Forest; GaussianNB", "decimal_places": "4",
     "rounding_evidence": "summary.round(4); HTML fractional values have four digits", "source_cell_index": source_cell},
    {"source": "lr_knn_cv_summary.csv", "models": "Logistic Regression; k-NN", "decimal_places": "stored CSV precision",
     "rounding_evidence": "No additional rounding before aggregation", "source_cell_index": None},
])
precision.to_csv(results_dir / "summary_source_precision.csv", index=False)
display(precision)
display(initial_times.drop(columns=["source", "source_decimal_places"]).round(6))
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, model in zip(axes.flat, MODELS):
    group = initial_times.loc[initial_times.model == model]
    ax.plot(group.k, group.total_with_MI_seconds_mean, marker="o", label="including MI")
    ax.plot(group.k, group.total_without_MI_seconds_mean, marker="s", label="excluding MI")
    ax.set(title=model, xlabel="original features kept", ylabel="mean seconds per fold")
    ax.set_xticks(K_VALUES)
    ax.legend()
    ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
'''),
    markdown("""
## 3. Repeated timing for the no-removal calculation

Some recorded fit/predict means differ by about 30–40% for the same calculation.
Percent differences below use (08 minus 06) divided by 06. Both tables use five folds.
"""),
    code(r'''
lof_summary = pd.read_csv(results_dir / "lof_cv_summary.csv")
iforest_summary = pd.read_csv(results_dir / "iforest_cv_summary.csv")
timing_fields = ["model", "fit_seconds_mean", "predict_seconds_mean"]
timing_variation = lof_summary.loc[lof_summary.condition == "No removal", timing_fields].merge(
    iforest_summary.loc[iforest_summary.condition == "No removal", timing_fields],
    on="model", suffixes=("_06", "_08"), validate="one_to_one"
)
for metric in ["fit", "predict"]:
    timing_variation[f"{metric}_difference_pct_vs_06"] = (
        (timing_variation[f"{metric}_seconds_mean_08"] - timing_variation[f"{metric}_seconds_mean_06"])
        / timing_variation[f"{metric}_seconds_mean_06"] * 100
    )
timing_variation = timing_variation.set_index("model").loc[MODELS].reset_index()
timing_variation.to_csv(results_dir / "summary_no_removal_timing.csv", index=False)
display(timing_variation.round(6))
'''),
    markdown("""
## 4. MI selection stability across folds

Read LR's saved feature lists; the MI data, folds, seed, and function are shared
by all four classifiers. For k=33, every feature is retained without MI ranking.
"""),
    code(r'''
saved_folds = pd.read_csv(results_dir / "lr_knn_cv_folds.csv")
selections = saved_folds.loc[saved_folds.model == "Logistic Regression", ["fold", "k", "selected_features"]].copy()
selected_sets = {}
for row in selections.itertuples():
    features = json.loads(row.selected_features)
    assert len(features) == row.k and len(set(features)) == row.k
    selected_sets[(row.fold, row.k)] = set(features)
assert len(selected_sets) == 35
full_mi = pd.read_csv(results_dir / "knn_final_mi.csv")
all_features = full_mi.feature.tolist()
assert len(all_features) == 33 and len(set(all_features)) == 33
stability_rows = []
frequency = pd.DataFrame(index=pd.Index(all_features, name="feature"))
for k in K_VALUES:
    sets = [selected_sets[(fold, k)] for fold in range(1, 6)]
    intersection, union = set.intersection(*sets), set.union(*sets)
    stability_rows.append({"k": k, "all_five_count": len(intersection), "any_fold_count": len(union),
                           "all_five_features": json.dumps(sorted(intersection)),
                           "any_fold_features": json.dumps(sorted(union))})
    frequency[f"k_{k}"] = [sum(feature in values for values in sets) for feature in all_features]
stability = pd.DataFrame(stability_rows)
selections.sort_values(["k", "fold"]).to_csv(results_dir / "summary_feature_selections.csv", index=False)
stability.to_csv(results_dir / "summary_mi_stability.csv", index=False)
frequency.to_csv(results_dir / "summary_mi_selection_frequency.csv")
display(stability[["k", "all_five_count", "any_fold_count"]])
with pd.option_context("display.max_rows", None):
    display(frequency)
stable17 = set.intersection(*(selected_sets[(fold, 17)] for fold in range(1, 6)))
full_top17 = set(full_mi.feature.head(17))
comparison17 = pd.DataFrame({
    "feature": all_features, "full_training_MI_rank": range(1, 34),
    "folds_selected_at_k17": frequency["k_17"].to_numpy(),
    "all_five_at_k17": [feature in stable17 for feature in all_features],
    "full_training_top17": [feature in full_top17 for feature in all_features],
})
comparison17.to_csv(results_dir / "summary_mi_k17_vs_full_training.csv", index=False)
display(comparison17.loc[comparison17.all_five_at_k17])
print("Shared stable/full-training top-17 features:", len(stable17 & full_top17))
print("Stable-only features:", sorted(stable17 - full_top17))
print("Full-training-top-17-only features:", sorted(full_top17 - stable17))
jaccard_rows = []
for k in [5, 17]:
    for a, b in combinations(range(1, 6), 2):
        left, right = selected_sets[(a, k)], selected_sets[(b, k)]
        overlap, union = len(left & right), len(left | right)
        jaccard_rows.append({"k": k, "fold_a": a, "fold_b": b, "intersection_count": overlap,
                             "union_count": union, "Jaccard": overlap / union})
jaccard_pairs = pd.DataFrame(jaccard_rows)
jaccard_summary = jaccard_pairs.groupby("k").agg(
    pair_count=("Jaccard", "size"), Jaccard_mean=("Jaccard", "mean"),
    Jaccard_min=("Jaccard", "min"), Jaccard_max=("Jaccard", "max")
).reset_index()
jaccard_pairs.to_csv(results_dir / "summary_mi_jaccard_pairs.csv", index=False)
jaccard_summary.to_csv(results_dir / "summary_mi_jaccard_summary.csv", index=False)
display(jaccard_pairs.round(6))
display(jaccard_summary.round(6))
'''),
    markdown("""
## 5. Saved k=4 cluster descriptions

Recreate only the shared sample order; do not fit preprocessing or clustering.
Numeric summaries use the seven original numeric features. Categorical summaries
use the full-training MI top-17 categorical features. Missing values count as a
category; modal ties are resolved lexically. Income is used for descriptions only.
"""),
    code(r'''
data = load_training_data(repo_root)
X_cluster, y_cluster = clustering_sample(data.X_dev, data.y_dev)
kmeans_assignments = pd.read_csv(results_dir / "kmeans_assignments.csv")
ward_assignments = pd.read_csv(results_dir / "ward_assignments.csv")
for frame in [kmeans_assignments, ward_assignments]:
    assert len(frame) == 10_000
    assert np.array_equal(frame.row_index.to_numpy(), X_cluster.index.to_numpy())
assignment_sources = {
    "K-Means": (kmeans_assignments, "cluster_k_"),
    "Ward": (ward_assignments, "ward_cluster_k_"),
}
categorical_features = [feature for feature in full_mi.feature.head(17) if feature not in NUMERIC_COLUMNS]
numeric_rows, categorical_rows = [], []
for method, (frame, prefix) in assignment_sources.items():
    labels = frame[f"{prefix}4"].to_numpy()
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        group = X_cluster.loc[mask]
        numeric_rows.append({
            "method": method, "k": 4, "cluster": int(cluster), "count": int(mask.sum()),
            "high_income_rate": float(y_cluster.iloc[mask].mean()),
            **{f"{column}_mean": float(group[column].mean()) for column in NUMERIC_COLUMNS},
        })
        for feature in categorical_features:
            counts = group[feature].fillna("Missing").astype(str).value_counts(dropna=False)
            mode_count = int(counts.max())
            mode = sorted(counts.index[counts == mode_count].tolist())[0]
            categorical_rows.append({"method": method, "k": 4, "cluster": int(cluster),
                                     "feature": feature, "mode": mode, "mode_count": mode_count,
                                     "mode_fraction": mode_count / len(group)})
cluster_numeric = pd.DataFrame(numeric_rows)
cluster_categorical = pd.DataFrame(categorical_rows)
cluster_numeric.to_csv(results_dir / "summary_cluster_numeric_k4.csv", index=False)
cluster_categorical.to_csv(results_dir / "summary_cluster_categorical_k4.csv", index=False)
with pd.option_context("display.max_columns", None, "display.max_rows", None):
    display(cluster_numeric.round(6))
    display(cluster_categorical.round(6))
cross = pd.crosstab(kmeans_assignments.cluster_k_4, ward_assignments.ward_cluster_k_4,
                    rownames=["kmeans_cluster"], colnames=["ward_cluster"])
cross.rename(columns=lambda value: f"ward_cluster_{value}").to_csv(
    results_dir / "summary_cluster_crosstab_k4.csv", index_label="kmeans_cluster"
)
display(cross)
'''),
    markdown("""
## 6. Income-label purity of saved clusters

Purity is the sum of each cluster's majority-class count divided by 10,000.
The one-cluster reference is the sample's low-income proportion, 0.9139.
No assignments are fitted or changed for this calculation.
"""),
    code(r'''
purity_rows = []
purity_baseline = float(y_cluster.eq(0).mean())
for method, (frame, prefix) in assignment_sources.items():
    for k in range(2, 11):
        labels = frame[f"{prefix}{k}"].to_numpy()
        assert len(np.unique(labels)) == k
        class_counts = pd.crosstab(labels, y_cluster.to_numpy()).reindex(columns=[0, 1], fill_value=0)
        majority_count = int(class_counts.max(axis=1).sum())
        purity_rows.append({"method": method, "k": k, "majority_correct_count": majority_count,
                            "sample_count": len(y_cluster), "purity": majority_count / len(y_cluster),
                            "single_cluster_baseline": purity_baseline,
                            "difference_from_baseline": majority_count / len(y_cluster) - purity_baseline})
purity = pd.DataFrame(purity_rows)
purity.to_csv(results_dir / "summary_cluster_purity.csv", index=False)
display(purity.round(6))
'''),
    markdown("## 7. Final checks and execution time"),
    code(r'''
assert len(initial_times) == 28
assert set(initial_times.model) == set(MODELS)
assert initial_times.groupby("model").k.apply(lambda values: set(values) == set(K_VALUES)).all()
assert initial_times.loc[initial_times.k == 33, "MI_seconds_mean"].eq(0).all()
assert np.allclose(initial_times.loc[initial_times.k == 33, "relative_to_k33"], 1)
assert np.allclose(initial_times.total_with_MI_seconds_mean, initial_times[TIME_COLUMNS].sum(axis=1))
assert len(timing_variation) == 4 and len(precision) == 2
assert len(selections) == 35 and len(selected_sets) == 35
assert set(selections.fold) == set(range(1, 6))
assert len(stability) == 7 and frequency.shape == (33, 7)
assert frequency.min().min() >= 0 and frequency.max().max() <= 5
assert frequency.k_33.eq(5).all()
assert (np.diff(frequency.to_numpy(), axis=1) >= 0).all()
assert stability.loc[stability.k == 33, "all_five_count"].iloc[0] == 33
assert len(jaccard_pairs) == 20 and jaccard_summary.pair_count.eq(10).all()
assert jaccard_pairs.Jaccard.between(0, 1).all()
assert len(data.X_dev) == 30_000 and len(data.folds) == 5 and X_cluster.shape == (10_000, 33)
assert len(cluster_numeric) == 8
assert len(cluster_categorical) == 8 * len(categorical_features)
assert len(NUMERIC_COLUMNS) == 7 and cluster_numeric.groupby("method")["count"].sum().eq(10_000).all()
assert cross.shape == (4, 4) and cross.to_numpy().sum() == 10_000
assert len(purity) == 18 and set(purity.k) == set(range(2, 11))
assert np.isclose(purity_baseline, 0.9139, rtol=0, atol=1e-12)
assert purity.purity.between(purity_baseline, 1).all()
for method, prefix in [("K-Means", "kmeans"), ("Ward", "ward")]:
    saved = pd.read_csv(results_dir / f"{prefix}_cluster_profiles.csv")
    current = cluster_numeric.loc[cluster_numeric.method == method].sort_values("cluster")
    reference = saved.loc[saved.k == 4].sort_values("cluster")
    assert np.array_equal(current["count"], reference["count"])
    assert np.allclose(current.high_income_rate, reference.high_income_rate, rtol=0, atol=1e-12)
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in input_hashes.items())
assert all(item["num_threads"] == 1 for item in threadpool_info())
total_seconds = time.perf_counter() - experiment_started
pd.DataFrame([{"total_seconds": total_seconds, "python": platform.python_version(),
               "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
               "native_threads": 1, "saved_03_decimal_places": 4}]).to_csv(
    results_dir / "summary_execution.csv", index=False
)
print("Final checks passed.")
print(f"Total execution time: {total_seconds:.2f} seconds")
'''),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {
    "display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells, start=1):
    cell["id"] = f"summary-{index:02d}"
path = Path(__file__).resolve().parents[1] / "notebooks/09_summary_analysis.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
print(f"Built {path}")
