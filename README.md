# CMPT455-project-Nagano

## EDA and preprocessing notebooks

Install the packages in `requirements.txt`, then open the notebooks from the repository root or the `notebooks/` directory in Colab, Kaggle, or Jupyter. The first code cell downloads and prepares the UCI Census-Income data if the processed CSV is absent. Keep the saved cell outputs when submitting the notebooks.

- `notebooks/01_eda.ipynb`: EDA contributed by the other developer.
- `notebooks/02_preprocessing.ipynb`: compares preprocessing options on training data only. It requires `src/data_loader.py` and `src/preprocessing.py`; `scripts/build_preprocessing_notebook.py` is the reproducible notebook source.

The 02 notebook holds out 20% of the rows for later evaluation. It compares 33 versus 37 original features, raw versus `log1p` monetary values, and class weighting by five-fold CV on a 30,000-row training sample. It also reports mutual-information rankings and exploratory PCA. The held-out test set is not evaluated.

## Classification comparison

Run `notebooks/04_classification_lr_knn.ipynb` top to bottom from the project root or `notebooks/` after installing `requirements.txt` (in Colab, clone/upload the repository and change to its root).
It compares LR and k-NN, refits selected models on the full training set, and saves CSV results in `results/`; keep the notebook outputs.
Regenerate it with `python scripts/build_lr_knn_notebook.py`, then rerun the notebook to restore outputs.

Run `notebooks/03_classification_comparison.ipynb` from top to bottom. It
compares Random Forest and Gaussian Naive Bayes on the same stratified
30,000-row training sample with five-fold cross-validation, then evaluates
the selected settings once on the held-out test set. Keep the saved outputs.

The English notebook is the project copy. Regenerate it with
`python scripts/build_classification_notebook.py` from the project root, then
rerun it to restore outputs. The optional Japanese study copy is generated only
when `scripts/classification_notebook_ja.json` is present.

## Clustering and outlier removal

Run `notebooks/05_kmeans.ipynb` first from the repository root or `notebooks/` after installing `requirements.txt`; in Colab, clone/upload the repository and change to its root.
Regenerate it with `python scripts/build_kmeans_notebook.py`; outputs are saved as `results/kmeans_*.csv`.
`COMMON_K` is 4, selected by the agreed rule in notebook 07; run notebook 05 before notebook 07.

Then run `notebooks/07_ward_clustering.ipynb` to fit Ward hierarchical clustering on the same 10,000 training-only sample and compare its scores with the saved K-Means results. It reports the best k for each method and metric, applies the pre-set common-k rule, and compares post-clustering income profiles at that shared k. The Ward notebook is generated with `python scripts/build_ward_notebook.py`; its outputs are saved as `results/ward_*.csv` and `results/clustering_*.csv`.

Run `notebooks/06_lof_outliers.ipynb` top to bottom in the same environment, retaining the existing classification CSVs for reference checks.
Regenerate it with `python scripts/build_lof_notebook.py`; outputs are saved as `results/lof_*.csv`. Both notebooks require `src/unsupervised_common.py` and retain their executed outputs.

Run `notebooks/08_isolation_forest_outliers.ipynb` after notebook 06. It applies Isolation Forest at 1% and 5% contamination using the same folds and four fixed classifiers, then combines its fold results with LOF in `results/outlier_*.csv`. Regenerate it with `python scripts/build_iforest_notebook.py`.

## Summary analysis

Run notebooks 03–08 first, then run `notebooks/09_summary_analysis.ipynb` top to bottom from the repository root or `notebooks/`.
Install `lxml` in addition to `requirements.txt`; regenerate 09 with `python scripts/build_summary_notebook.py`.
It reads saved results and the shared training sample without fitting models, and saves tables as `results/summary_*.csv`.

## Training size and feature selection

Run `notebooks/10_training_size_feature_selection.ipynb` top to bottom from the repository root or `notebooks/` after installing `requirements.txt`.
Regenerate it with `python scripts/build_training_size_notebook.py`; results and checkpoints are saved as `results/trainsize_*.csv`.
Allow roughly 20–40 minutes for all 120 single-threaded runs on this machine; the fixed 20,000-row validation set comes only from the original training split.
