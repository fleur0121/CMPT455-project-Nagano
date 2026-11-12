"""Build the reproducible Random Forest / GaussianNB comparison notebook.

Run from the project root with ``python scripts/build_classification_notebook.py``.
The script uses only Python's standard library to build the notebook JSON.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


cells = [
    markdown(r"""
# RF and GaussianNB comparison

Use a stratified 30,000-row sample from the training split for 5-fold cross-validation and parameter selection. Keep the selected feature count and best parameters, then refit each final model on all 172,459 training rows. For k < 33, rank features by mutual information on all training rows. Evaluate the 43,115-row held-out test set once per model.

Average precision (AP) is the main metric. ROC-AUC, accuracy, precision, recall, F1, and processing times are also recorded. Test results are exploratory and are not used to change settings.
"""),
    markdown("## 1. Imports, data, and shared split"),
    code(r"""
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
get_ipython().run_line_magic("matplotlib", "inline")
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder

cwd = Path.cwd()
repo_root = cwd if (cwd / "src" / "preprocessing.py").exists() else cwd.parent
assert (repo_root / "src" / "preprocessing.py").exists(), (
    "Open this notebook from the project root or notebooks/ directory."
)
sys.path.insert(0, str(repo_root / "src"))
from preprocessing import NUMERIC_COLUMNS, build_preprocessor, prepare_xy, select_features

data_path = repo_root / "data" / "processed" / "census_income_adults.csv"
if not data_path.exists():
    import subprocess
    subprocess.run([sys.executable, "src/data_loader.py"], cwd=repo_root, check=True)

df = pd.read_csv(data_path, skipinitialspace=True)
X, y = prepare_xy(df)
X = select_features(X, include_migration=False)  # Baseline: 33 original features

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
print(f"Original features: {X.shape[1]}")
print(f"Training rows: {len(X_train):,}; held-out test rows: {len(X_test):,}")
print(f"CV sample rows: {len(X_dev):,}; positive rate: {y_dev.mean():.4f}")
print(f"CV folds: {len(folds)}; feature counts: {FEATURE_COUNTS}")
"""),
    markdown(r"""
## 2. MI ranking within each training fold

MI ranks the original features using only the training part of each fold. The validation rows are not used to select features.
"""),
    code(r"""
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

# Small preview; the actual CV ranks features within each fold.
display(rank_features_by_mi(X_dev.iloc[:1_000], y_dev.iloc[:1_000]).head(10))
"""),
    markdown("## 3. Evaluation functions and initial model comparison"),
    code(r"""
def make_model(model_name, params=None):
    params = {} if params is None else dict(params)
    if model_name == "Random Forest":
        settings = {"n_estimators": 200, "random_state": 42, "n_jobs": 1}
        settings.update(params)
        return RandomForestClassifier(**settings)
    if model_name == "GaussianNB":
        settings = {"var_smoothing": 1e-9}
        settings.update(params)
        return GaussianNB(**settings)
    raise ValueError(f"Unknown model: {model_name}")


def run_cv(model_name, k, params=None):
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
        if model_name == "GaussianNB":
            if hasattr(Z_fit, "toarray"):
                Z_fit, Z_valid = Z_fit.toarray(), Z_valid.toarray()
        preprocessing_seconds = time.perf_counter() - started

        model = make_model(model_name, params)
        started = time.perf_counter()
        model.fit(Z_fit, y_fit)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        probability = model.predict_proba(Z_valid)[:, 1]
        predict_seconds = time.perf_counter() - started
        predicted = (probability >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_valid, predicted, labels=[0, 1]
        ).ravel()

        rows.append({
            "model": model_name, "k": k, "fold": fold_number,
            "AP": average_precision_score(y_valid, probability),
            "ROC_AUC": roc_auc_score(y_valid, probability),
            "Accuracy": accuracy_score(y_valid, predicted),
            "Precision": precision_score(y_valid, predicted, zero_division=0),
            "Recall": recall_score(y_valid, predicted, zero_division=0),
            "F1": f1_score(y_valid, predicted, zero_division=0),
            "MI_seconds": mi_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
    return pd.DataFrame(rows)


all_folds = []
for model_name in ["Random Forest", "GaussianNB"]:
    for k in FEATURE_COUNTS:
        print(f"Running {model_name}, {k} features ...", flush=True)
        all_folds.append(run_cv(model_name, k))

fold_results = pd.concat(all_folds, ignore_index=True)
summary = (
    fold_results.groupby(["model", "k"])
    .agg(
        AP_mean=("AP", "mean"), AP_sd=("AP", "std"),
        ROC_AUC_mean=("ROC_AUC", "mean"),
        Accuracy_mean=("Accuracy", "mean"),
        Precision_mean=("Precision", "mean"),
        Recall_mean=("Recall", "mean"), F1_mean=("F1", "mean"),
        MI_seconds_mean=("MI_seconds", "mean"),
        preprocessing_seconds_mean=("preprocessing_seconds", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
        predict_seconds_mean=("predict_seconds", "mean"),
    )
    .reset_index()
)
display(summary.round(4))
"""),
    markdown("## 4. Compare AP and computation time"),
    code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for model_name, group in summary.groupby("model"):
    group = group.sort_values("k")
    axes[0].errorbar(group["k"], group["AP_mean"], yerr=group["AP_sd"], marker="o", capsize=3, label=model_name)
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
"""),
    markdown(r"""
## 5. Hyperparameter search

The parameter search compares a small set of settings on the same folds. The table below shows the tested settings and scores.
"""),
    code(r"""
chosen_k = (
    summary.sort_values(["model", "AP_mean", "k"], ascending=[True, False, True])
    .groupby("model", as_index=False).first()[["model", "k"]]
    .set_index("model")["k"].astype(int).to_dict()
)
print("Feature count chosen from initial CV AP:", chosen_k)

parameter_grids = {
    "Random Forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 5],
    },
    "GaussianNB": {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
}

tuning_rows = []
for model_name, grid in parameter_grids.items():
    for params in ParameterGrid(grid):
        print(f"Tuning {model_name}: {params}", flush=True)
        scores = run_cv(model_name, chosen_k[model_name], params=params)
        tuning_rows.append({
            "model": model_name,
            "k": chosen_k[model_name],
            "params": params,
            "AP_mean": scores["AP"].mean(),
            "AP_sd": scores["AP"].std(),
            "ROC_AUC_mean": scores["ROC_AUC"].mean(),
            "F1_mean": scores["F1"].mean(),
            "fit_seconds_mean": scores["fit_seconds"].mean(),
        })

tuning_summary = pd.DataFrame(tuning_rows)
best_tuning = (
    tuning_summary.sort_values(["model", "AP_mean"], ascending=[True, False])
    .groupby("model", as_index=False).first()
)
display(tuning_summary.sort_values(["model", "AP_mean"], ascending=[True, False]).round(4))
print("Selected settings by CV AP:")
display(best_tuning[["model", "k", "params", "AP_mean", "AP_sd"]])
"""),
    markdown(r"""
## 6. One-time held-out test evaluation

After selecting the settings, each model is refit on all 172,459 training rows and evaluated once on the 43,115-row held-out test set. For k < 33, MI ranking uses all training rows. The test results are not used to change the settings. The confusion matrices use a 0.5 threshold.
"""),
    code(r"""
final_rows = []
final_predictions = {}
final_models = {}
final_fit_rows = {}

assert len(X_train) == 172_459
assert len(X_test) == 43_115

for _, choice in best_tuning.iterrows():
    model_name = choice["model"]
    k = int(choice["k"])
    params = choice["params"]

    # Reuse CV-selected k and parameters. Rank on all training rows for k < 33.
    if k == X_train.shape[1]:
        selected = list(X_train.columns)
    else:
        selected = rank_features_by_mi(X_train, y_train)["feature"].head(k).tolist()

    X_train_k, X_test_k = X_train[selected], X_test[selected]
    preprocessor = build_preprocessor(X_train_k, log_money=False)
    Z_train = preprocessor.fit_transform(X_train_k)
    Z_test = preprocessor.transform(X_test_k)
    if model_name == "GaussianNB" and hasattr(Z_train, "toarray"):
        Z_train, Z_test = Z_train.toarray(), Z_test.toarray()

    model = make_model(model_name, params)
    model.fit(Z_train, y_train)
    probability = model.predict_proba(Z_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    final_models[model_name] = (model, preprocessor, selected)
    final_predictions[model_name] = (probability, predicted)
    final_fit_rows[model_name] = len(X_train)
    final_rows.append({
        "model": model_name, "k": k, "params": params,
        "training_rows": len(X_train), "test_rows": len(X_test),
        "AP": average_precision_score(y_test, probability),
        "ROC_AUC": roc_auc_score(y_test, probability),
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1": f1_score(y_test, predicted, zero_division=0),
    })

test_summary = pd.DataFrame(final_rows)
display(test_summary.round(4))
"""),
    markdown("## 7. Confusion matrices and ROC curves"),
    code(r"""
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for row, model_name in enumerate(["Random Forest", "GaussianNB"]):
    probability, predicted = final_predictions[model_name]
    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted, labels=[0, 1],
        display_labels=["<=50K", ">50K"],
        cmap="Blues", values_format="d", ax=axes[row, 0],
    )
    axes[row, 0].set_title(f"{model_name}: held-out confusion matrix (threshold=0.5)")
    RocCurveDisplay.from_predictions(y_test, probability, name=model_name, ax=axes[row, 1])
    axes[row, 1].plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axes[row, 1].set_title(f"{model_name}: held-out ROC curve")
    axes[row, 1].set_xlabel("False Positive Rate")
    axes[row, 1].set_ylabel("True Positive Rate")
fig.tight_layout()
plt.show()
"""),
    markdown(r"""
## 8. Final checks
"""),
    code(r"""
assert len(X_train) == 172_459
assert len(X_test) == 43_115
assert X.shape[1] == 33
assert len(X_dev) == 30_000
assert len(folds) == 5
assert len(fold_results) == 60
assert fold_results.groupby(["model", "k"]).size().eq(5).all()
assert len(tuning_summary) == 12
assert set(test_summary["model"]) == {"Random Forest", "GaussianNB"}
assert test_summary["training_rows"].eq(172_459).all()
assert test_summary["test_rows"].eq(43_115).all()
assert final_fit_rows == {"Random Forest": 172_459, "GaussianNB": 172_459}
assert all(len(final_predictions[name][0]) == 43_115 for name in final_predictions)
print("Final checks passed.")
print(f"CV result rows: {len(fold_results):,} (expected 60)")
print(f"Tuning configurations: {len(tuning_summary):,} (expected 12)")
print(f"Final fit rows per model: {final_fit_rows}")
print(f"Held-out test rows per model: {len(X_test):,}")
"""),
    markdown(""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for index, cell in enumerate(notebook["cells"], start=1):
    cell["id"] = f"classification-{index:02d}"

translation_path = Path(__file__).with_name("classification_notebook_ja.json")
output_dir = Path(__file__).resolve().parents[1] / "notebooks"
output_dir.mkdir(parents=True, exist_ok=True)
outputs = [
    (output_dir / "03_classification_comparison.ipynb", notebook),
]
if translation_path.exists():
    translations = json.loads(translation_path.read_text())
    japanese_notebook = deepcopy(notebook)

    for index, source in translations["markdown"].items():
        japanese_notebook["cells"][int(index)]["source"] = source.splitlines(keepends=True)

    for cell in japanese_notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for old, new_text in translations["code_replacements"].items():
            source = source.replace(old, new_text)
        cell["source"] = source.splitlines(keepends=True)

    imports_cell = japanese_notebook["cells"][2]
    source = "".join(imports_cell["source"])
    source = source.replace(
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.pyplot as plt\nfrom matplotlib import font_manager\n",
    )
    source = source.replace(
        'get_ipython().run_line_magic("matplotlib", "inline")\n',
        'get_ipython().run_line_magic("matplotlib", "inline")\n'
        + translations["japanese_font_setup"],
    )
    imports_cell["source"] = source.splitlines(keepends=True)
    outputs.append((output_dir / "03_classification_comparison_ja.ipynb", japanese_notebook))
else:
    print("Translation file not found; building the English notebook only.")

for output_path, content in outputs:
    output_path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    print(f"Built {output_path}")
