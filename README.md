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

`notebooks/03_classification_comparison_ja.ipynb` is a Japanese study copy.
The English notebook is the project copy. To regenerate both notebooks, run
`python scripts/build_classification_notebook.py` from the project root, then
run the notebooks again to restore their outputs.
