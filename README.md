# CMPT455-project-Nagano

## EDA and preprocessing notebooks

Install the packages in `requirements.txt`, then open the notebooks from the repository root or the `notebooks/` directory in Colab, Kaggle, or Jupyter. The first code cell downloads and prepares the UCI Census-Income data if the processed CSV is absent. Keep the saved cell outputs when submitting the notebooks.

- `notebooks/01_eda.ipynb`: EDA contributed by the other developer.
- `notebooks/02_preprocessing.ipynb`: compares preprocessing options on training data only. It requires `src/data_loader.py` and `src/preprocessing.py`; `scripts/build_preprocessing_notebook.py` is the reproducible notebook source.

The 02 notebook holds out 20% of the rows for later evaluation. It compares 33 versus 37 original features, raw versus `log1p` monetary values, and class weighting by five-fold CV on a 30,000-row training sample. It also reports mutual-information rankings and exploratory PCA. The held-out test set is not evaluated.
