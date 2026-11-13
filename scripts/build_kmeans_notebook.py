"""Build the reproducible training-only K-Means notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.strip().splitlines(keepends=True)}


cells = [
    markdown("""
# K-Means clustering

Use a stratified 10,000-row sample from the shared 30,000 training rows.
Fit the shared 33-feature preprocessing on these 10,000 rows. Clustering and
all three scores use the full one-hot space; PCA is for visualization only.
Income labels enter stratified sampling and post-fit profiles only.
No test rows are used. Silhouette is computed on all 10,000 rows.
"""),
    markdown("## 1. Environment and training data"),
    code(r'''
from pathlib import Path
import sys
import time
import platform

experiment_started = time.perf_counter()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from IPython.display import display
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_info, threadpool_limits
get_ipython().run_line_magic("matplotlib", "inline")
thread_limit = threadpool_limits(limits=1)
cwd = Path.cwd()
repo_root = cwd if (cwd / "src/preprocessing.py").exists() else cwd.parent
assert (repo_root / "src/unsupervised_common.py").exists(), "Open from the repository root or notebooks/."
sys.path.insert(0, str(repo_root / "src"))
from unsupervised_common import (
    load_training_data, clustering_sample, prepare_clustering,
    evaluate_clustering, best_cluster_counts, pca_projection,
)
results_dir = repo_root / "results"
results_dir.mkdir(exist_ok=True)
print(f"Python {platform.python_version()}; sklearn {sklearn.__version__}; pandas {pd.__version__}; numpy {np.__version__}")
data = load_training_data(repo_root)
print(f"Training rows: {len(data.X_train):,}; CV sample: {len(data.X_dev):,}; folds: {len(data.folds)}")
'''),
    markdown("""
## 2. Clustering sample and preprocessing

Stratified class counts are integers. The rate check allows one observation
of rounding error in 10,000 rows. The dense matrix retains every one-hot column
and can also be used for Ward linkage.
"""),
    code(r'''
X_cluster, y_cluster = clustering_sample(data.X_dev, data.y_dev)
preprocessor, Z_cluster, preprocessing_seconds = prepare_clustering(X_cluster)
sample_info = pd.DataFrame([{
    "dev_rows": len(data.X_dev), "cluster_rows": len(X_cluster),
    "original_features": X_cluster.shape[1], "encoded_features": Z_cluster.shape[1],
    "dev_high_income_rate": data.y_dev.mean(), "sample_high_income_rate": y_cluster.mean(),
    "rate_tolerance": 1 / len(X_cluster), "preprocessing_seconds": preprocessing_seconds,
}])
sample_info.to_csv(results_dir / "kmeans_sample_info.csv", index=False)
display(sample_info)
'''),
    markdown("## 3. Cluster counts and full-sample scores"),
    code(r'''
K_VALUES = list(range(2, 11))
def make_kmeans(k):
    return KMeans(n_clusters=k, n_init=10, random_state=42)

scores, profiles, assignments = evaluate_clustering(
    Z_cluster, y_cluster, make_kmeans, K_VALUES
)
best_k = best_cluster_counts(scores)
scores.to_csv(results_dir / "kmeans_scores.csv", index=False)
best_k.to_csv(results_dir / "kmeans_best_k.csv", index=False)
profiles.to_csv(results_dir / "kmeans_cluster_profiles.csv", index=False)
membership = pd.DataFrame({"row_index": X_cluster.index})
for k in K_VALUES:
    membership[f"cluster_k_{k}"] = assignments[k]
membership.to_csv(results_dir / "kmeans_assignments.csv", index=False)
display(scores.round(6))
display(best_k)
'''),
    markdown("## 4. Score plots and post-fit income profiles"),
    code(r'''
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, metric in zip(axes, ["Silhouette", "Calinski_Harabasz", "Davies_Bouldin"]):
    ax.plot(scores["k"], scores[metric], marker="o")
    ax.set(title=metric.replace("_", " "), xlabel="number of clusters", ylabel=metric)
    ax.set_xticks(K_VALUES)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
with pd.option_context("display.max_rows", None):
    display(profiles.round(6))
'''),
    markdown("## 5. PCA visualization at the Silhouette-selected k"),
    code(r'''
coordinates, pca = pca_projection(Z_cluster)
pd.DataFrame({"component": [1, 2], "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(
    results_dir / "kmeans_pca_variance.csv", index=False
)
def plot_clusters(k, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=assignments[k],
                        cmap="tab10", s=7, alpha=0.6, vmin=-0.5, vmax=k - 0.5)
    fig.colorbar(points, ax=ax, ticks=range(k), label="cluster")
    ax.set(title=title, xlabel="PC1", ylabel="PC2")
    fig.tight_layout()
    plt.show()
silhouette_k = int(best_k.loc[best_k.metric == "Silhouette", "k"].iloc[0])
plot_clusters(silhouette_k, f"K-Means: k={silhouette_k}; PCA visualization only")
'''),
    markdown("## 6. Optional common k for comparison with Ward linkage"),
    code(r'''
COMMON_K = None
if COMMON_K is None:
    print("Common k is not yet decided.")
else:
    assert COMMON_K in K_VALUES
    display(profiles.loc[profiles.k == COMMON_K].round(6))
    plot_clusters(COMMON_K, f"K-Means: common k={COMMON_K}; PCA visualization only")
'''),
    markdown("## 7. Final checks and execution time"),
    code(r'''
assert len(data.X_dev) == 30_000 and len(data.folds) == 5
assert X_cluster.shape == (10_000, 33)
assert np.isclose(y_cluster.mean(), data.y_dev.mean(), atol=1 / 10_000, rtol=0)
assert scores.k.tolist() == list(range(2, 11)) and len(scores) == 9
assert np.isfinite(scores[["Silhouette", "Calinski_Harabasz", "Davies_Bouldin"]]).all().all()
assert len(profiles) == sum(K_VALUES)
assert profiles.groupby("k")["count"].sum().eq(10_000).all()
assert all(len(np.unique(assignments[k])) == k for k in K_VALUES)
assert set(X_cluster.index).issubset(set(data.X_dev.index))
assert all(item["num_threads"] == 1 for item in threadpool_info())
assert COMMON_K is None or COMMON_K in K_VALUES
total_seconds = time.perf_counter() - experiment_started
pd.DataFrame([{"total_seconds": total_seconds, "python": platform.python_version(),
               "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
               "native_threads": 1}]).to_csv(results_dir / "kmeans_execution.csv", index=False)
print("Final checks passed.")
print(f"Total execution time: {total_seconds:.2f} seconds")
'''),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {
    "display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells, start=1):
    cell["id"] = f"kmeans-{index:02d}"
path = Path(__file__).resolve().parents[1] / "notebooks/05_kmeans.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
print(f"Built {path}")
