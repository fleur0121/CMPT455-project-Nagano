"""Training-only utilities shared by clustering and outlier experiments."""

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, calinski_harabasz_score,
    confusion_matrix, davies_bouldin_score, f1_score, precision_score,
    recall_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder

from preprocessing import MONEY_COLUMNS, NUMERIC_COLUMNS, build_preprocessor, prepare_xy, select_features


MODEL_SPECS = {
    "Random Forest": {"k": 33, "params": {"n_estimators": 400, "max_depth": 20,
        "min_samples_leaf": 1, "random_state": 42, "n_jobs": 1}},
    "GaussianNB": {"k": 5, "params": {"var_smoothing": 1e-6}},
    "Logistic Regression": {"k": 17, "params": {"C": 0.1, "random_state": 42, "max_iter": 100}},
    "k-NN": {"k": 17, "params": {"n_neighbors": 51, "weights": "uniform", "n_jobs": 1}},
}
METRICS = ["AP", "ROC_AUC", "Accuracy", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP"]
TIMINGS = ["detector_preprocessing_seconds", "detector_fit_seconds", "MI_seconds",
           "preprocessing_seconds", "fit_seconds", "predict_seconds"]
WARNING_COLUMNS = ["condition", "fold", "phase", "model", "category", "message"]


@dataclass
class TrainingData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_dev: pd.DataFrame
    y_dev: pd.Series
    folds: list


def load_training_data(repo_root):
    """Reproduce notebooks 03/04 without returning or evaluating test rows."""
    root = Path(repo_root)
    path = root / "data/processed/census_income_adults.csv"
    if not path.exists():
        subprocess.run([sys.executable, "src/data_loader.py"], cwd=root, check=True)
    df = pd.read_csv(path, skipinitialspace=True)
    X, y = prepare_xy(df)
    X = select_features(X, include_migration=False)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_dev, _, y_dev, _ = train_test_split(
        X_train, y_train, train_size=30_000, stratify=y_train, random_state=42
    )
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_dev, y_dev))
    assert X.shape[1] == 33
    assert len(X_train) == 172_459 and len(X_dev) == 30_000 and len(folds) == 5
    return TrainingData(X_train, y_train, X_dev, y_dev, folds)


def clustering_sample(X_dev, y_dev):
    """Use income for stratified extraction only; retain labels for profiles."""
    X_sample, _, y_sample, _ = train_test_split(
        X_dev, y_dev, train_size=10_000, stratify=y_dev, random_state=42
    )
    assert len(X_sample) == 10_000 and X_sample.index.equals(y_sample.index)
    # Integer class counts allow at most one observation of rounding error.
    assert np.isclose(y_sample.mean(), y_dev.mean(), atol=1 / len(y_sample), rtol=0)
    return X_sample, y_sample


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


def prepare_clustering(X_sample):
    """Return the full one-hot space, also usable by dense-only Ward linkage."""
    preprocessor = build_preprocessor(X_sample, log_money=False)
    started = time.perf_counter()
    Z = preprocessor.fit_transform(X_sample)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return preprocessor, Z, time.perf_counter() - started


def cluster_profiles(labels, y_sample, k):
    return (
        pd.DataFrame({"cluster": labels, "high_income": np.asarray(y_sample)})
        .groupby("cluster").agg(count=("high_income", "size"),
                                high_income_rate=("high_income", "mean"))
        .reset_index().assign(k=k)[["k", "cluster", "count", "high_income_rate"]]
    )


def evaluate_clustering(Z, y_sample, clusterer_factory, cluster_counts=range(2, 11)):
    """Inject a factory taking k; labels never enter fit_predict or the scores."""
    rows, profiles, assignments = [], [], {}
    for k in cluster_counts:
        estimator = clusterer_factory(k)
        started = time.perf_counter()
        labels = estimator.fit_predict(Z)
        fit_seconds = time.perf_counter() - started
        assert len(np.unique(labels)) == k
        scores, durations = {}, {}
        for name, score in [("Silhouette", silhouette_score),
                            ("Calinski_Harabasz", calinski_harabasz_score),
                            ("Davies_Bouldin", davies_bouldin_score)]:
            started = time.perf_counter()
            scores[name] = score(Z, labels)
            durations[name + "_seconds"] = time.perf_counter() - started
        rows.append({"k": k, **scores, "fit_seconds": fit_seconds,
                     "metrics_seconds": sum(durations.values()), **durations})
        assignments[k] = labels
        profiles.append(cluster_profiles(labels, y_sample, k))
        print(f"Completed clustering k={k}", flush=True)
    return pd.DataFrame(rows), pd.concat(profiles, ignore_index=True), assignments


def best_cluster_counts(scores):
    rows = []
    for metric, ascending in [("Silhouette", False), ("Calinski_Harabasz", False),
                               ("Davies_Bouldin", True)]:
        best = scores.sort_values([metric, "k"], ascending=[ascending, True]).iloc[0]
        rows.append({"metric": metric, "direction": "min" if ascending else "max",
                     "k": int(best["k"]), "value": best[metric]})
    return pd.DataFrame(rows)


def pca_projection(Z):
    """Fit a two-dimensional visualization; do not use its output for models."""
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(Z), pca


def make_classifier(model_name):
    classes = {"Random Forest": RandomForestClassifier, "GaussianNB": GaussianNB,
               "Logistic Regression": LogisticRegression, "k-NN": KNeighborsClassifier}
    return classes[model_name](**MODEL_SPECS[model_name]["params"])


def classification_metrics(y_true, probability):
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {"AP": average_precision_score(y_true, probability),
            "ROC_AUC": roc_auc_score(y_true, probability),
            "Accuracy": accuracy_score(y_true, predicted),
            "Precision": precision_score(y_true, predicted, zero_division=0),
            "Recall": recall_score(y_true, predicted, zero_division=0),
            "F1": f1_score(y_true, predicted, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def _record_warnings(caught, records, condition, fold, phase, model):
    for item in caught:
        records.append({"condition": condition, "fold": fold, "phase": phase,
                        "model": model, "category": item.category.__name__,
                        "message": str(item.message)})


def run_outlier_cv(data, detector_factory, contamination, condition,
                   model_names=None):
    """Fit a detector only on training rows; keep every validation row.

    detector_factory(contamination) must provide fit_predict(X), with -1 for
    outliers. The same per-fold mask is shared by all classifiers. Warnings are
    returned for explicit audit; model settings are never changed in response.
    """
    names = list(MODEL_SPECS) if model_names is None else list(model_names)
    rows, removals, money, members, warning_rows = [], [], [], [], []
    first_fold = None
    for fold_number, (fit_idx, valid_idx) in enumerate(data.folds, start=1):
        X_fit, y_fit = data.X_dev.iloc[fit_idx], data.y_dev.iloc[fit_idx]
        X_valid, y_valid = data.X_dev.iloc[valid_idx], data.y_dev.iloc[valid_idx]
        keep = np.ones(len(X_fit), dtype=bool)
        detector_preprocessing_seconds, detector_fit_seconds = 0.0, 0.0
        if contamination is not None:
            detector_preprocessor = build_preprocessor(X_fit, log_money=False)
            started = time.perf_counter()
            Z_detector = detector_preprocessor.fit_transform(X_fit)
            detector_preprocessing_seconds = time.perf_counter() - started
            detector = detector_factory(contamination)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                flags = np.asarray(detector.fit_predict(Z_detector))
            detector_fit_seconds = time.perf_counter() - started
            _record_warnings(caught, warning_rows, condition, fold_number, "detector", "")
            assert set(np.unique(flags)).issubset({-1, 1})
            keep = flags != -1
            if fold_number == 1:
                first_fold = {"Z": Z_detector, "removed": ~keep,
                              "row_index": X_fit.index.to_numpy()}
        removed_count = int((~keep).sum())
        removal = {"condition": condition, "fold": fold_number,
                   "contamination": contamination, "training_count": len(X_fit),
                   "retained_count": int(keep.sum()), "removed_count": removed_count,
                   "removed_fraction": removed_count / len(X_fit),
                   "training_high_income_rate": float(y_fit.mean()),
                   "removed_high_income_rate": float(y_fit.iloc[~keep].mean()) if removed_count else np.nan,
                   "retained_high_income_rate": float(y_fit.iloc[keep].mean()),
                   "detector_preprocessing_seconds": detector_preprocessing_seconds,
                   "detector_fit_seconds": detector_fit_seconds}
        removals.append(removal)
        if contamination is not None:
            members.extend({"condition": condition, "fold": fold_number,
                            "row_index": int(index)} for index in X_fit.index[~keep])
            for group, mask in [("removed", ~keep), ("retained", keep)]:
                for column in MONEY_COLUMNS:
                    values = X_fit.loc[mask, column]
                    stats = values.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
                    money.append({"condition": condition, "fold": fold_number,
                                  "group": group, "feature": column, **stats})
        X_retained, y_retained = X_fit.iloc[keep], y_fit.iloc[keep]
        assert y_retained.nunique() == 2
        for model_name in names:
            k = MODEL_SPECS[model_name]["k"]
            mi_seconds = 0.0
            if k == X_retained.shape[1]:
                selected = list(X_retained.columns)
            else:
                started = time.perf_counter()
                ranking = rank_features_by_mi(X_retained, y_retained)
                selected = ranking["feature"].head(k).tolist()
                mi_seconds = time.perf_counter() - started
            X_fit_k, X_valid_k = X_retained[selected], X_valid[selected]
            preprocessor = build_preprocessor(X_fit_k, log_money=False)
            started = time.perf_counter()
            Z_fit = preprocessor.fit_transform(X_fit_k)
            Z_valid = preprocessor.transform(X_valid_k)
            if model_name == "GaussianNB" and hasattr(Z_fit, "toarray"):
                Z_fit, Z_valid = Z_fit.toarray(), Z_valid.toarray()
            preprocessing_seconds = time.perf_counter() - started
            model = make_classifier(model_name)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(Z_fit, y_retained)
            fit_seconds = time.perf_counter() - started
            _record_warnings(caught, warning_rows, condition, fold_number, "classifier", model_name)
            started = time.perf_counter()
            probability = model.predict_proba(Z_valid)[:, 1]
            predict_seconds = time.perf_counter() - started
            rows.append({**removal, "model": model_name, "k": k,
                         "validation_count": len(X_valid), **classification_metrics(y_valid, probability),
                         "MI_seconds": mi_seconds, "preprocessing_seconds": preprocessing_seconds,
                         "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                         "selected_features": json.dumps(selected)})
        print(f"Completed {condition}, fold {fold_number}/5", flush=True)
    return {"folds": pd.DataFrame(rows), "removals": pd.DataFrame(removals),
            "money": pd.DataFrame(money), "members": pd.DataFrame(members),
            "warnings": pd.DataFrame(warning_rows, columns=WARNING_COLUMNS), "first_fold": first_fold}


def summarize_outlier_results(fold_results):
    columns = METRICS + TIMINGS
    result = fold_results.groupby(["condition", "model"])[columns].agg(["mean", "std"])
    result.columns = [f"{metric}_{'sd' if statistic == 'std' else statistic}"
                      for metric, statistic in result.columns]
    return result.reset_index()


def paired_outlier_differences(fold_results, baseline_condition="No removal"):
    baseline = fold_results.loc[fold_results.condition == baseline_condition,
                                ["model", "fold"] + METRICS]
    paired = fold_results.merge(baseline, on=["model", "fold"], suffixes=("", "_baseline"), validate="many_to_one")
    for metric in METRICS:
        paired[metric + "_difference"] = paired[metric] - paired[metric + "_baseline"]
    columns = [metric + "_difference" for metric in METRICS]
    summary = paired.groupby(["condition", "model"])[columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{'sd' if statistic == 'std' else statistic}"
                       for metric, statistic in summary.columns]
    return paired[["condition", "model", "fold"] + columns], summary.reset_index()


def check_no_removal_baseline(fold_results, repo_root):
    current = fold_results.loc[fold_results.condition == "No removal"]
    saved = pd.read_csv(Path(repo_root) / "results/lr_knn_tuning_folds.csv")
    rows = []
    for model_name, params in [("Logistic Regression", {"C": 0.1}),
                                ("k-NN", {"n_neighbors": 51, "weights": "uniform"})]:
        reference = saved.loc[(saved.model == model_name) & (saved.k == 17)
                              & saved.params.map(lambda value: json.loads(value) == params)]
        assert len(reference) == 5
        for fold in range(1, 6):
            actual = float(current.loc[(current.model == model_name) & (current.fold == fold), "AP"].iloc[0])
            expected = float(reference.loc[reference.fold == fold, "AP"].iloc[0])
            rows.append({"model": model_name, "fold": fold, "actual_AP": actual,
                         "reference_AP": expected, "absolute_difference": abs(actual - expected),
                         "tolerance": 1e-12, "matches": np.isclose(actual, expected, atol=1e-12, rtol=0)})
    for model_name, expected in [("Random Forest", 0.6238), ("GaussianNB", 0.2241)]:
        actual = float(current.loc[current.model == model_name, "AP"].mean())
        rows.append({"model": model_name, "fold": "mean", "actual_AP": actual,
                     "reference_AP": expected, "absolute_difference": abs(actual - expected),
                     "tolerance": 5e-5, "matches": np.isclose(actual, expected, atol=5e-5, rtol=0)})
    return pd.DataFrame(rows)
