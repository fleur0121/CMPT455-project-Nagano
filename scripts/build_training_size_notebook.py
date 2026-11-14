"""Build the training-size and MI-cost experiment notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.strip().splitlines(keepends=True)}


cells = [
    markdown("""
# Training size and feature-selection cost

Hold out 20,000 validation rows from the original training split. Draw stratified
training samples of 5,000, 10,000, 30,000, 60,000, and 120,000 rows using seeds 0–2.
Use the fixed MODEL_SPECS parameters with 17 or 33 original features for every model.
The original test rows are never returned or evaluated.

MI is computed once per sample and its measured time is charged to each k=17 run.
Each classifier refits its own preprocessing. Predictions use all 20,000 validation
rows at threshold 0.5. Totals include MI, preprocessing, fit, and predict_proba.
"""),
    markdown("## 1. Environment and training-only validation split"),
    code(r'''
from pathlib import Path
from itertools import product
import hashlib
import json
import platform
import sys
import time
import warnings

experiment_started = time.perf_counter()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from IPython.display import display
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_info, threadpool_limits
get_ipython().run_line_magic("matplotlib", "inline")
thread_limit = threadpool_limits(limits=1)
cwd = Path.cwd()
repo_root = cwd if (cwd / "src/unsupervised_common.py").exists() else cwd.parent
assert (repo_root / "src/unsupervised_common.py").exists(), "Open from the repository root or notebooks/."
sys.path.insert(0, str(repo_root / "src"))
from preprocessing import build_preprocessor
from unsupervised_common import (
    MODEL_SPECS, load_training_data, rank_features_by_mi,
    make_classifier, classification_metrics,
)
results_dir = repo_root / "results"
results_dir.mkdir(exist_ok=True)
N_VALUES = [5_000, 10_000, 30_000, 60_000, 120_000]
SEEDS = [0, 1, 2]
FEATURE_COUNTS = [17, 33]
MODELS = list(MODEL_SPECS)
TIME_COLUMNS = ["MI_seconds", "preprocessing_seconds", "fit_seconds", "predict_seconds", "total_seconds"]
protected_paths = [p for folder in ["notebooks", "scripts", "src", "results"]
                   for p in (repo_root / folder).iterdir() if p.is_file()
                   and not p.name.startswith(("10_training_size", "build_training_size", "trainsize_"))]
protected_paths.append(repo_root / "requirements.txt")
input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths}
print(f"Python {platform.python_version()}; sklearn {sklearn.__version__}; pandas {pd.__version__}; numpy {np.__version__}")
data = load_training_data(repo_root)
X_pool, X_valid, y_pool, y_valid = train_test_split(
    data.X_train, data.y_train, test_size=20_000, stratify=data.y_train, random_state=42
)
training_ids = set(data.X_train.index)
validation_ids = set(X_valid.index)
assert len(data.X_train) == 172_459 and len(X_pool) == 152_459 and len(X_valid) == 20_000
assert set(X_pool.index).isdisjoint(validation_ids)
assert set(X_pool.index) | validation_ids == training_ids
assert validation_ids.issubset(training_ids)
assert data.X_train.shape[1] == 33
validation_index_sha256 = hashlib.sha256(X_valid.index.to_numpy(dtype=np.int64).tobytes()).hexdigest()
print(f"Training pool: {len(X_pool):,}; fixed validation rows: {len(X_valid):,}")
display(pd.DataFrame([{"model": name, "params": json.dumps(spec["params"]),
                       "feature_counts": str(FEATURE_COUNTS)} for name, spec in MODEL_SPECS.items()]))
'''),
    markdown("""
## 2. Fit policy and checkpoints

LR starts at max_iter=100 for each run. As in notebook 04, a convergence warning
triggers a fresh fit with twice the limit, up to 25,600. Fit time includes all
attempts. Captured warnings and iteration limits are saved without changing
other parameters. One progress line is printed per completed condition.
Membership in the canonical X_train excludes the original test rows.
"""),
    code(r'''
records, sample_records, ranking_records = [], [], []
fit_attempts, warning_records = [], []
WARNING_COLUMNS = ["N", "random_state", "k", "model", "max_iter", "category", "message"]

def checkpoint():
    if records:
        pd.DataFrame(records).to_csv(results_dir / "trainsize_runs.csv", index=False)
    if sample_records:
        pd.DataFrame(sample_records).to_csv(results_dir / "trainsize_samples.csv", index=False)
    if ranking_records:
        pd.concat(ranking_records, ignore_index=True).to_csv(results_dir / "trainsize_mi_rankings.csv", index=False)
    if fit_attempts:
        pd.DataFrame(fit_attempts).to_csv(results_dir / "trainsize_fit_attempts.csv", index=False)
    pd.DataFrame(warning_records, columns=WARNING_COLUMNS).to_csv(
        results_dir / "trainsize_warnings.csv", index=False
    )

def fit_with_retries(model_name, Z_fit, y_fit, context):
    model = make_classifier(model_name)
    started = time.perf_counter()
    attempts, convergence_count, warning_count = 0, 0, 0
    while True:
        attempt_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(Z_fit, y_fit)
        attempt_seconds = time.perf_counter() - attempt_started
        limit = model.get_params().get("max_iter")
        convergence = any(issubclass(item.category, ConvergenceWarning) for item in caught)
        attempts += 1
        convergence_count += int(convergence)
        warning_count += len(caught)
        fit_attempts.append({**context, "attempt": attempts, "max_iter": limit,
                             "seconds": attempt_seconds, "convergence_warning": convergence})
        for item in caught:
            warning_records.append({**context, "max_iter": limit,
                                    "category": item.category.__name__, "message": str(item.message)})
        if not convergence:
            break
        if model_name != "Logistic Regression" or limit >= 25_600:
            checkpoint()
            raise RuntimeError(f"Unresolved convergence warning: {context}")
        model = make_classifier(model_name)
        model.set_params(max_iter=limit * 2)
    return model, time.perf_counter() - started, attempts, convergence_count, warning_count
'''),
    markdown("## 3. Runs in increasing training-size order"),
    code(r'''
for N in N_VALUES:
    for seed in SEEDS:
        X_fit, _, y_fit, _ = train_test_split(
            X_pool, y_pool, train_size=N, stratify=y_pool, random_state=seed
        )
        fit_ids = set(X_fit.index)
        assert len(X_fit) == N and X_fit.index.is_unique
        assert fit_ids.issubset(training_ids) and fit_ids.isdisjoint(validation_ids)
        assert np.isclose(y_fit.mean(), y_pool.mean(), atol=1 / N, rtol=0)
        started = time.perf_counter()
        ranking = rank_features_by_mi(X_fit, y_fit)
        selected17 = ranking["feature"].head(17).tolist()
        sample_MI_seconds = time.perf_counter() - started
        sample_records.append({
            "N": N, "random_state": seed, "sample_count": len(X_fit),
            "sample_positive_count": int(y_fit.sum()), "sample_positive_rate": y_fit.mean(),
            "pool_positive_rate": y_pool.mean(), "rate_tolerance": 1 / N,
            "validation_count": len(X_valid), "validation_overlap": len(fit_ids & validation_ids),
            "training_membership_verified": fit_ids.issubset(training_ids),
            "sample_index_sha256": hashlib.sha256(X_fit.index.to_numpy(dtype=np.int64).tobytes()).hexdigest(),
            "validation_index_sha256": validation_index_sha256,
            "MI_seconds": sample_MI_seconds, "selected_features_k17": json.dumps(selected17),
        })
        ranking_records.append(ranking.assign(N=N, random_state=seed, rank=range(1, 34)))
        checkpoint()
        for k in FEATURE_COUNTS:
            selected = selected17 if k == 17 else list(X_fit.columns)
            mi_seconds = sample_MI_seconds if k == 17 else 0.0
            for model_name in MODELS:
                X_fit_k, X_valid_k = X_fit[selected], X_valid[selected]
                preprocessor = build_preprocessor(X_fit_k, log_money=False)
                started = time.perf_counter()
                Z_fit = preprocessor.fit_transform(X_fit_k)
                Z_valid = preprocessor.transform(X_valid_k)
                if model_name == "GaussianNB" and hasattr(Z_fit, "toarray"):
                    Z_fit, Z_valid = Z_fit.toarray(), Z_valid.toarray()
                preprocessing_seconds = time.perf_counter() - started
                context = {"N": N, "random_state": seed, "k": k, "model": model_name}
                model, fit_seconds, attempts, convergence_count, warning_count = fit_with_retries(
                    model_name, Z_fit, y_fit, context
                )
                started = time.perf_counter()
                probability = model.predict_proba(Z_valid)[:, 1]
                predict_seconds = time.perf_counter() - started
                total_seconds = mi_seconds + preprocessing_seconds + fit_seconds + predict_seconds
                records.append({
                    **context, "training_count": len(X_fit), "validation_count": len(X_valid),
                    "sample_positive_rate": y_fit.mean(), "validation_positive_rate": y_valid.mean(),
                    "MI_seconds": mi_seconds, "preprocessing_seconds": preprocessing_seconds,
                    "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                    "total_seconds": total_seconds, **classification_metrics(y_valid, probability),
                    "encoded_features": Z_fit.shape[1], "selected_features": json.dumps(selected),
                    "max_iter": model.get_params().get("max_iter"), "fit_attempts": attempts,
                    "convergence_warning_count": convergence_count, "fit_warning_count": warning_count,
                    "validation_index_sha256": validation_index_sha256,
                })
                checkpoint()
                print(f"[{len(records)}/120] N={N}, seed={seed}, k={k}, {model_name}: "
                      f"AP={records[-1]['AP']:.6f}, total={total_seconds:.3f}s, "
                      f"max_iter={records[-1]['max_iter']}, attempts={attempts}", flush=True)
runs = pd.DataFrame(records)
samples = pd.DataFrame(sample_records)
checkpoint()
print(f"Captured fit diagnostics: {len(warning_records)}")
'''),
    markdown("""
## 4. Three-seed summaries and paired time ratios

Each ratio is computed within the same training size, sample seed, and model
before aggregation. MI-to-fit/predict divides k=17 MI time by its fit plus
predict time. Strict comparisons to 1 classify the three paired ratios.
"""),
    code(r'''
summary_columns = TIME_COLUMNS + ["AP", "ROC_AUC", "F1"]
summary = runs.groupby(["N", "k", "model"])[summary_columns].agg(["mean", "min", "max"])
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary = summary.reset_index()
summary.to_csv(results_dir / "trainsize_summary.csv", index=False)
pair_columns = ["N", "random_state", "model", "total_seconds", "MI_seconds", "fit_seconds", "predict_seconds"]
paired = runs.loc[runs.k == 17, pair_columns].merge(
    runs.loc[runs.k == 33, pair_columns], on=["N", "random_state", "model"],
    suffixes=("_17", "_33"), validate="one_to_one"
)
paired["total_ratio"] = paired.total_seconds_17 / paired.total_seconds_33
paired["MI_over_fit_predict_ratio"] = paired.MI_seconds_17 / (paired.fit_seconds_17 + paired.predict_seconds_17)
paired.to_csv(results_dir / "trainsize_paired_ratios.csv", index=False)
ratio = paired.groupby(["N", "model"])[["total_ratio", "MI_over_fit_predict_ratio"]].agg(["mean", "min", "max"])
ratio.columns = [f"{metric}_{stat}" for metric, stat in ratio.columns]
ratio = ratio.reset_index()
status_rows = []
for (N, model_name), group in paired.groupby(["N", "model"]):
    values = group.total_ratio.to_numpy()
    status = ("all below 1" if (values < 1).all() else
              "all above 1" if (values > 1).all() else "mixed or equal")
    status_rows.append({"N": N, "model": model_name, "repetitions": len(group),
                        "ratio_status": status, "ratios_below_1": int((values < 1).sum()),
                        "ratios_above_1": int((values > 1).sum()), "ratios_equal_1": int((values == 1).sum())})
ratio = ratio.merge(pd.DataFrame(status_rows), on=["N", "model"], validate="one_to_one")
ratio.to_csv(results_dir / "trainsize_ratio.csv", index=False)
mi_summary = samples.groupby("N").agg(MI_seconds_mean=("MI_seconds", "mean"),
                                      MI_seconds_min=("MI_seconds", "min"),
                                      MI_seconds_max=("MI_seconds", "max"),
                                      repetitions=("MI_seconds", "size")).reset_index()
mi_summary.to_csv(results_dir / "trainsize_mi_summary.csv", index=False)
display(summary.round(6))
display(ratio.round(6))
display(mi_summary.round(6))
'''),
    markdown("## 5. Total time, paired ratios, and validation AP"),
    code(r'''
for model_name in MODELS:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for k in FEATURE_COUNTS:
        group = summary.loc[(summary.model == model_name) & (summary.k == k)].sort_values("N")
        for ax, metric in [(axes[0], "total_seconds"), (axes[2], "AP")]:
            ax.plot(group.N, group[f"{metric}_mean"], marker="o", label=f"k={k}")
            ax.fill_between(group.N, group[f"{metric}_min"], group[f"{metric}_max"], alpha=0.15)
    group = ratio.loc[ratio.model == model_name].sort_values("N")
    axes[1].plot(group.N, group.total_ratio_mean, marker="o", label="k=17 / k=33")
    axes[1].fill_between(group.N, group.total_ratio_min, group.total_ratio_max, alpha=0.15)
    axes[1].axhline(1, color="grey", linestyle="--", linewidth=1)
    axes[0].set(title="Total time including MI", ylabel="seconds")
    axes[1].set(title="Paired total-time ratio", ylabel="ratio")
    axes[2].set(title="Fixed validation AP", ylabel="AP")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(N_VALUES, [f"{n:,}" for n in N_VALUES])
        ax.set_xlabel("training rows N")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(model_name)
    fig.tight_layout()
    plt.show()
'''),
    markdown("## 6. Final checks and execution record"),
    code(r'''
expected = set(product(N_VALUES, SEEDS, FEATURE_COUNTS, MODELS))
actual = set(runs[["N", "random_state", "k", "model"]].itertuples(index=False, name=None))
assert len(runs) == 120 and actual == expected
assert not runs.duplicated(["N", "random_state", "k", "model"]).any()
assert runs.validation_count.eq(20_000).all()
assert runs.validation_index_sha256.nunique() == 1
assert runs.training_count.eq(runs.N).all()
assert runs.loc[runs.k == 33, "MI_seconds"].eq(0).all()
assert runs.loc[runs.k == 17].groupby(["N", "random_state"]).MI_seconds.nunique().eq(1).all()
assert np.allclose(runs.total_seconds, runs[TIME_COLUMNS[:-1]].sum(axis=1))
assert runs[["TN", "FP", "FN", "TP"]].sum(axis=1).eq(20_000).all()
assert len(samples) == 15 and len(pd.concat(ranking_records)) == 495
assert samples.validation_overlap.eq(0).all() and samples.training_membership_verified.all()
assert np.all(np.abs(samples.sample_positive_rate - samples.pool_positive_rate) <= samples.rate_tolerance)
assert len(summary) == 40 and len(paired) == 60 and len(ratio) == 20
assert ratio.repetitions.eq(3).all() and mi_summary.repetitions.eq(3).all()
assert all(item["num_threads"] == 1 for item in threadpool_info())
assert make_classifier("Random Forest").get_params()["n_jobs"] == 1
assert make_classifier("k-NN").get_params()["n_jobs"] == 1
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in input_hashes.items())
checkpoint()
execution_seconds = time.perf_counter() - experiment_started
lr_adjusted = runs.loc[(runs.model == "Logistic Regression") & (runs.max_iter > 100)]
pd.DataFrame([{"total_seconds": execution_seconds, "completed_runs": len(runs),
               "captured_fit_diagnostics": len(warning_records), "LR_adjusted_conditions": len(lr_adjusted),
               "python": platform.python_version(), "sklearn": sklearn.__version__,
               "pandas": pd.__version__, "numpy": np.__version__, "native_threads": 1}]).to_csv(
    results_dir / "trainsize_execution.csv", index=False
)
display(lr_adjusted[["N", "random_state", "k", "max_iter", "fit_attempts", "convergence_warning_count"]])
print("Final checks passed.")
print(f"Captured fit diagnostics: {len(warning_records)}")
print(f"Total execution time: {execution_seconds:.2f} seconds")
'''),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {
    "display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells, start=1):
    cell["id"] = f"training-size-{index:02d}"
path = Path(__file__).resolve().parents[1] / "notebooks/10_training_size_feature_selection.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
print(f"Built {path}")
