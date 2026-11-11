"""Build the standalone LR / k-NN experiment notebook using standard-library JSON."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.strip().splitlines(keepends=True)}


cells = [
    markdown("""
# Logistic Regression and k-NN comparison

The split, five folds, MI ranking, preprocessing, and metrics match notebook 03.
AP selects feature counts and parameters. Final refits use all 172,459 training
rows. The held-out test set is evaluated once per selected model.

Run from the repository root or `notebooks/` after installing `requirements.txt`.
In Colab, clone/upload the repository, install its requirements, and change the
working directory to the repository root before running these cells.
CSV outputs are saved in `results/`. No class weights or log transform are used.
Notebook 02 reports LR AP 0.6048 without weighting and 0.5975 with balanced weights.
"""),
    markdown("## 1. Imports, data, and shared split"),
    code(r'''
from pathlib import Path
import sys
import time
import json
import platform
import warnings

experiment_started = time.perf_counter()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from IPython.display import display
get_ipython().run_line_magic("matplotlib", "inline")
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits

# Limit native thread pools to one thread; k-NN also uses n_jobs=1.
thread_limit = threadpool_limits(limits=1)
cwd = Path.cwd()
repo_root = cwd if (cwd / "src" / "preprocessing.py").exists() else cwd.parent
assert (repo_root / "src" / "preprocessing.py").exists(), (
    "Open this notebook from the project root or notebooks/ directory."
)
sys.path.insert(0, str(repo_root / "src"))
from preprocessing import NUMERIC_COLUMNS, build_preprocessor, prepare_xy, select_features

results_dir = repo_root / "results"
results_dir.mkdir(exist_ok=True)
data_path = repo_root / "data" / "processed" / "census_income_adults.csv"
if not data_path.exists():
    import subprocess
    subprocess.run([sys.executable, "src/data_loader.py"], cwd=repo_root, check=True)
df = pd.read_csv(data_path, skipinitialspace=True)
X, y = prepare_xy(df)
X = select_features(X, include_migration=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_dev, _, y_dev, _ = train_test_split(
    X_train, y_train, train_size=30_000, stratify=y_train, random_state=42
)
folds = list(StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
).split(X_dev, y_dev))
FEATURE_COUNTS = [5, 10, 15, 20, 25, 33]
MODEL_NAMES = ["Logistic Regression", "k-NN"]
print(f"Python {platform.python_version()}; sklearn {sklearn.__version__}; pandas {pd.__version__}")
print(f"Original features: {X.shape[1]}")
print(f"Training rows: {len(X_train):,}; held-out test rows: {len(X_test):,}")
print(f"CV sample rows: {len(X_dev):,}; positive rate: {y_dev.mean():.4f}")
print(f"CV folds: {len(folds)}; feature counts: {FEATURE_COUNTS}")
'''),
    markdown("## 2. MI ranking within each training fold"),
    code(r'''
def rank_features_by_mi(X_fit, y_fit):
    # Rank original features using only this fold's training rows.
    work = X_fit.copy()
    numeric = [column for column in NUMERIC_COLUMNS if column in work.columns]
    categorical = [column for column in work.columns if column not in numeric]

    if numeric:
        work[numeric] = SimpleImputer(strategy="median").fit_transform(work[numeric])
    if categorical:
        encoded = SimpleImputer(
            strategy="constant", fill_value="Missing"
        ).fit_transform(work[categorical])
        encoded = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit_transform(encoded)
        work[categorical] = encoded

    discrete = [column in categorical for column in work.columns]
    scores = mutual_info_classif(
        work, y_fit, discrete_features=discrete,
        random_state=42, n_neighbors=3,
    )
    return (
        pd.DataFrame({"feature": work.columns, "MI": scores})
        .sort_values(["MI", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
'''),
    markdown("""
## 3. Evaluation functions and initial model comparison

Models use sklearn defaults, with `random_state=42` for LR and `n_jobs=1` for k-NN.
Native thread pools are limited to one thread. LR uses `max_iter=100`, except
for `C=10`: the previous run required `max_iter=200` in all five folds, so this
setting starts at 200. This preserves the previous effective iteration limits.
Unexpected convergence failures trigger a fresh fit with twice the iteration
limit, up to 25,600. Attempts are recorded; fit time includes any retries.
No test scores control iteration limits.
Prediction time measures `predict_proba`; preprocessing time includes both transforms.
"""),
    code(r'''
warning_rows = []
fit_attempt_rows = []

def save_records():
    pd.DataFrame(warning_rows, columns=["context", "category", "message", "max_iter"]).to_csv(
        results_dir / "lr_knn_warnings.csv", index=False
    )
    pd.DataFrame(fit_attempt_rows).to_csv(results_dir / "lr_knn_fit_attempts.csv", index=False)

def make_model(model_name, params=None):
    settings = {} if params is None else dict(params)
    if model_name == "Logistic Regression":
        if settings.get("C", 1.0) == 10:
            settings.setdefault("max_iter", 200)
        return LogisticRegression(random_state=42, **settings)
    if model_name == "k-NN":
        return KNeighborsClassifier(n_jobs=1, **settings)
    raise ValueError(f"Unknown model: {model_name}")

def fit_model(model_name, params, Z_fit, y_fit, context):
    model = make_model(model_name, params)
    started = time.perf_counter()
    while True:
        attempt_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(Z_fit, y_fit)
        attempt_seconds = time.perf_counter() - attempt_started
        limit = model.get_params().get("max_iter")
        convergence = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        fit_attempt_rows.append({"context": context, "model": model_name,
                                 "max_iter": limit, "seconds": attempt_seconds,
                                 "convergence_warning": convergence})
        for warning in caught:
            warning_rows.append({"context": context, "category": warning.category.__name__,
                                 "message": str(warning.message), "max_iter": limit})
            print(f"{context}: {warning.category.__name__}, max_iter={limit}: {warning.message}", flush=True)
        if not convergence:
            break
        if model_name != "Logistic Regression" or limit >= 25_600:
            save_records()
            raise RuntimeError(f"Unresolved convergence warning: {context}")
        settings = dict(params or {})
        settings["max_iter"] = limit * 2
        print(f"Retrying {context} with max_iter={settings['max_iter']}", flush=True)
        model = make_model(model_name, settings)
    return model, time.perf_counter() - started

def metrics(y_true, probability):
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "AP": average_precision_score(y_true, probability),
        "ROC_AUC": roc_auc_score(y_true, probability),
        "Accuracy": accuracy_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }

def run_cv(model_name, k, params=None, stage="initial"):
    rows = []
    for fold_number, (train_idx, valid_idx) in enumerate(folds, start=1):
        X_fit, y_fit = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_valid, y_valid = X_dev.iloc[valid_idx], y_dev.iloc[valid_idx]
        mi_seconds = 0.0
        if k == X_dev.shape[1]:
            selected = list(X_fit.columns)
        else:
            started = time.perf_counter()
            ranking = rank_features_by_mi(X_fit, y_fit)
            selected = ranking["feature"].head(k).tolist()
            mi_seconds = time.perf_counter() - started
        X_fit_k, X_valid_k = X_fit[selected], X_valid[selected]
        preprocessor = build_preprocessor(X_fit_k, log_money=False)
        started = time.perf_counter()
        Z_fit = preprocessor.fit_transform(X_fit_k)
        Z_valid = preprocessor.transform(X_valid_k)
        preprocessing_seconds = time.perf_counter() - started
        context = f"{stage}/{model_name}/k={k}/fold={fold_number}/params={params}"
        model, fit_seconds = fit_model(model_name, params, Z_fit, y_fit, context)
        started = time.perf_counter()
        probability = model.predict_proba(Z_valid)[:, 1]
        predict_seconds = time.perf_counter() - started
        rows.append({
            "model": model_name, "k": k, "fold": fold_number,
            **metrics(y_valid, probability),
            "MI_seconds": mi_seconds, "preprocessing_seconds": preprocessing_seconds,
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "max_iter": model.get_params().get("max_iter"),
            "selected_features": json.dumps(selected),
        })
    save_records()
    return pd.DataFrame(rows)

all_folds = []
for model_name in MODEL_NAMES:
    for k in FEATURE_COUNTS:
        print(f"Running {model_name}, {k} features ...", flush=True)
        all_folds.append(run_cv(model_name, k))
        pd.concat(all_folds, ignore_index=True).to_csv(results_dir / "lr_knn_cv_folds.csv", index=False)
fold_results = pd.concat(all_folds, ignore_index=True)
summary = (
    fold_results.groupby(["model", "k"])
    .agg(
        AP_mean=("AP", "mean"), AP_sd=("AP", "std"),
        ROC_AUC_mean=("ROC_AUC", "mean"), Accuracy_mean=("Accuracy", "mean"),
        Precision_mean=("Precision", "mean"), Recall_mean=("Recall", "mean"),
        F1_mean=("F1", "mean"), MI_seconds_mean=("MI_seconds", "mean"),
        preprocessing_seconds_mean=("preprocessing_seconds", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"), predict_seconds_mean=("predict_seconds", "mean"),
    ).reset_index()
)
summary.to_csv(results_dir / "lr_knn_cv_summary.csv", index=False)
display(summary.round(4))
'''),
    markdown("## 4. Compare AP and computation time"),
    code(r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for model_name, group in summary.groupby("model"):
    group = group.sort_values("k")
    axes[0].errorbar(group["k"], group["AP_mean"], yerr=group["AP_sd"],
                     marker="o", capsize=3, label=model_name)
    axes[1].plot(group["k"], group["fit_seconds_mean"], marker="o", label=model_name)
    axes[2].plot(group["k"], group["predict_seconds_mean"], marker="o", label=model_name)
axes[0].set(title="AP by feature count", xlabel="features kept", ylabel="mean AP ± SD")
axes[1].set(title="Fit time by feature count", xlabel="features kept", ylabel="seconds")
axes[2].set(title="Prediction time by feature count", xlabel="features kept", ylabel="seconds")
for ax in axes:
    ax.legend()
    ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
'''),
    markdown("""
## 5. Hyperparameter search

Each model keeps the feature count selected by initial CV AP (smaller k breaks ties).
ParameterGrid enumerates settings on the same five folds. The first tested setting
wins an AP tie. The comparison table uses CV results at the selected k.
"""),
    code(r'''
chosen_k = (
    summary.sort_values(["model", "AP_mean", "k"], ascending=[True, False, True])
    .groupby("model", as_index=False).first()[["model", "k"]]
    .set_index("model")["k"].astype(int).to_dict()
)
print("Feature count chosen from initial CV AP:", chosen_k)
parameter_grids = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
    "k-NN": {"n_neighbors": [5, 15, 31, 51, 71, 101, 151], "weights": ["uniform", "distance"]},
}
display(pd.DataFrame([
    {"model": name, "parameter": key, "search_range": values}
    for name, grid in parameter_grids.items() for key, values in grid.items()
]))
tuning_rows, tuning_folds = [], []
for model_name, grid in parameter_grids.items():
    for candidate, params in enumerate(ParameterGrid(grid)):
        print(f"Tuning {model_name}, k={chosen_k[model_name]}: {params}", flush=True)
        scores = run_cv(model_name, chosen_k[model_name], params, stage="tuning")
        tuning_folds.append(scores.assign(candidate=candidate, params=json.dumps(params)))
        tuning_rows.append({
            "model": model_name, "k": chosen_k[model_name], "candidate": candidate,
            "params": params, "AP_mean": scores["AP"].mean(), "AP_sd": scores["AP"].std(),
            **{f"{metric}_mean": scores[metric].mean() for metric in [
                "ROC_AUC", "Accuracy", "Precision", "Recall", "F1", "MI_seconds",
                "preprocessing_seconds", "fit_seconds", "predict_seconds"]},
        })
        pd.DataFrame(tuning_rows).assign(
            params=lambda frame: frame["params"].map(json.dumps)
        ).to_csv(results_dir / "lr_knn_tuning.csv", index=False)
        pd.concat(tuning_folds, ignore_index=True).to_csv(
            results_dir / "lr_knn_tuning_folds.csv", index=False
        )
tuning_summary = pd.DataFrame(tuning_rows)
best_tuning = (
    tuning_summary.sort_values(["model", "AP_mean"], ascending=[True, False])
    .groupby("model", as_index=False).first()
)
display(tuning_summary.round(4))
display(best_tuning[["model", "k", "params", "AP_mean", "AP_sd"]])
knn_best = best_tuning.loc[best_tuning["model"] == "k-NN", "params"].iloc[0]
knn_at_upper_bound = knn_best["n_neighbors"] == max(parameter_grids["k-NN"]["n_neighbors"])
print(f"Best k-NN n_neighbors: {knn_best['n_neighbors']}; at upper bound: {knn_at_upper_bound}")
initial_chosen = summary[summary.apply(lambda row: row["k"] == chosen_k[row["model"]], axis=1)]
comparison_columns = ["model", "k", "setting", "AP_mean", "ROC_AUC_mean", "F1_mean",
                      "fit_seconds_mean", "predict_seconds_mean"]
comparison = pd.concat([initial_chosen.assign(setting="initial"),
                        best_tuning.assign(setting="best")], ignore_index=True)[comparison_columns]
display(comparison.round(4))
comparison.to_csv(results_dir / "lr_knn_initial_vs_best.csv", index=False)
'''),
    markdown("""
## 6. One-time held-out test evaluation

Each selected model is refit on all 172,459 training rows. MI uses these training
rows only and is skipped for k=33. Test scores do not change any settings.
Prediction progress is checkpointed before evaluation. An interrupted prediction
is recorded without assigning a test score.
"""),
    code(r'''
assert len(X_train) == 172_459
final_rows, final_predictions, final_models = [], {}, {}
for _, choice in best_tuning.iterrows():
    model_name, k, params = choice["model"], int(choice["k"]), choice["params"]
    print(f"Final refit: {model_name}, k={k}, params={params}, rows={len(X_train)}", flush=True)
    mi_seconds = 0.0
    if k == X_train.shape[1]:
        selected = list(X_train.columns)
    else:
        started = time.perf_counter()
        ranking = rank_features_by_mi(X_train, y_train)
        selected = ranking["feature"].head(k).tolist()
        mi_seconds = time.perf_counter() - started
        ranking.to_csv(results_dir / ("lr_final_mi.csv" if model_name == "Logistic Regression"
                                     else "knn_final_mi.csv"), index=False)
    X_train_k, X_test_k = X_train[selected], X_test[selected]
    preprocessor = build_preprocessor(X_train_k, log_money=False)
    started = time.perf_counter()
    Z_train = preprocessor.fit_transform(X_train_k)
    Z_test = preprocessor.transform(X_test_k)
    preprocessing_seconds = time.perf_counter() - started
    model, fit_seconds = fit_model(model_name, params, Z_train, y_train, f"final/{model_name}")
    row = {"model": model_name, "k": k, "params": json.dumps(params),
           "train_rows": len(X_train), "test_rows": len(X_test),
           "MI_seconds": mi_seconds, "preprocessing_seconds": preprocessing_seconds,
           "fit_seconds": fit_seconds, "max_iter": model.get_params().get("max_iter"),
           "selected_features": json.dumps(selected), "status": "predicting"}
    final_rows.append(row)
    pd.DataFrame(final_rows).to_csv(results_dir / "lr_knn_test_results.csv", index=False)
    print(f"Predicting {len(X_test):,} test rows with {model_name} ...", flush=True)
    started = time.perf_counter()
    try:
        probability = model.predict_proba(Z_test)[:, 1]
    except (KeyboardInterrupt, MemoryError) as error:
        row.update(status=type(error).__name__, predict_seconds=time.perf_counter() - started)
        print(f"Prediction incomplete: {row}", flush=True)
        pd.DataFrame(final_rows).to_csv(results_dir / "lr_knn_test_results.csv", index=False)
        continue
    row.update(predict_seconds=time.perf_counter() - started, status="complete")
    row.update(metrics(y_test, probability))
    final_predictions[model_name] = (probability, (probability >= 0.5).astype(int))
    final_models[model_name] = (model, preprocessor, selected)
    pd.DataFrame(final_rows).to_csv(results_dir / "lr_knn_test_results.csv", index=False)
    save_records()
test_summary = pd.DataFrame(final_rows)
display(test_summary.drop(columns="selected_features").round(4))
'''),
    markdown("## 7. Confusion matrices and ROC curves"),
    code(r'''
for model_name in MODEL_NAMES:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if model_name not in final_predictions:
        for ax in axes:
            ax.text(0.5, 0.5, f"{model_name}: prediction incomplete", ha="center")
            ax.set_axis_off()
        plt.show()
        continue
    probability, predicted = final_predictions[model_name]
    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted, labels=[0, 1], display_labels=["<=50K", ">50K"],
        cmap="Blues", values_format="d", ax=axes[0],
    )
    axes[0].set_title(f"{model_name}: test confusion matrix (threshold=0.5)")
    RocCurveDisplay.from_predictions(y_test, probability, name=model_name, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axes[1].set_title(f"{model_name}: held-out ROC curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    fig.tight_layout()
    plt.show()
'''),
    markdown("## 8. Final checks and execution record"),
    code(r'''
assert len(X_dev) == 30_000
assert X.shape[1] == 33
assert len(folds) == 5
assert len(fold_results) == 60
assert fold_results.groupby(["model", "k"]).size().eq(5).all()
assert fold_results.loc[fold_results["k"] == 33, "MI_seconds"].eq(0).all()
assert len(tuning_summary) == 18
assert set(test_summary["model"]) == set(MODEL_NAMES)
save_records()
attempts = pd.DataFrame(fit_attempt_rows)
print("LR iteration limits used:", sorted(attempts["max_iter"].dropna().unique().astype(int)))
display(attempts.groupby(["model", "max_iter"], dropna=False).agg(
    attempts=("seconds", "size"), convergence_warnings=("convergence_warning", "sum")
).reset_index())
total_seconds = time.perf_counter() - experiment_started
execution_record = pd.DataFrame([{
    "total_seconds": total_seconds, "warnings": len(warning_rows),
    "python": platform.python_version(), "sklearn": sklearn.__version__,
    "pandas": pd.__version__, "numpy": np.__version__, "n_jobs": 1,
    "completed_test_models": int(test_summary["status"].eq("complete").sum()),
}])
execution_record.to_csv(results_dir / "lr_knn_execution.csv", index=False)
display(execution_record)
print("Final checks passed.")
print(f"Total execution time: {total_seconds:.2f} seconds")
'''),
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
for index, cell in enumerate(cells, start=1):
    cell["id"] = f"lr-knn-{index:02d}"
output = Path(__file__).resolve().parents[1] / "notebooks" / "04_classification_lr_knn.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
print(f"Built {output}")
