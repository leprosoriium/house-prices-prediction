# House Prices Prediction

**Predicting residential sale prices with regression models and scikit-learn pipelines.**

[![Kaggle](https://img.shields.io/badge/Kaggle-House%20Prices-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Task](https://img.shields.io/badge/Task-Regression-informational)

## Project Overview

This project investigates residential property-price prediction using the Kaggle **House Prices: Advanced Regression Techniques** dataset. It covers EDA, log transformations, missing-value handling, feature screening, pipeline-based preprocessing, model comparison, cross-validation, hyperparameter tuning, and a Kaggle submission.

**Reported Kaggle score: 0.14772 RMSLE.** This score was supplied by the project author; it is not embedded in the notebook output.

## Results

| Model | Mean CV RMSE (log price) | CV MAE | CV R² |
|---|---:|---:|---:|
| Gradient Boosting | **0.1306** | 0.0916 | 0.8838 |
| Lasso | 0.1306 | 0.0943 | 0.8845 |
| XGBoost | 0.1309 | 0.0931 | 0.8830 |
| Ridge | 0.1310 | 0.0946 | 0.8840 |
| Extra Trees | 0.1384 | 0.0973 | 0.8693 |
| Random Forest | 0.1391 | 0.0959 | 0.8694 |

- **Baseline Linear Regression holdout RMSE:** 0.13317 (log price)
- **Final XGBoost holdout RMSE:** 0.12611 (log price)
- **Reported Kaggle leaderboard RMSLE:** **0.14772**

Gradient Boosting had the lowest mean CV RMSE at the displayed precision, while **XGBoost was chosen for the final submission**. The notebook does not establish that XGBoost was the best CV model.

## Workflow

1. **Data inspection:** feature types, duplicates, missing values, and summary statistics.
2. **EDA:** log-price distribution, numerical correlations, categorical ANOVA, and outlier inspection.
3. **Feature preparation:** log transforms, categorical missing-value semantics, and feature screening.
4. **Preprocessing:** numerical imputation/scaling, ordinal encoding, and one-hot encoding via `ColumnTransformer`.
5. **Validation:** 80/20 train/holdout split and shuffled 12-fold KFold CV.
6. **Models:** Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, Extra Trees, and XGBoost.
7. **Tuning:** `GridSearchCV` minimizing RMSE on `log1p(SalePrice)`.
8. **Submission:** tuned XGBoost, inverse log transform, `Id` / `SalePrice` CSV.

## Reproducibility

The notebook was developed for Kaggle and reads competition data from:

```text
/kaggle/input/competitions/house-prices-advanced-regression-techniques/train.csv
/kaggle/input/competitions/house-prices-advanced-regression-techniques/test.csv
```

1. Accept the [Kaggle competition rules](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
2. Open [`house-prices-prediction.ipynb`](house-prices-prediction.ipynb) in a Kaggle notebook with the competition data attached.
3. Run cells in order. The full XGBoost grid evaluates **3,072 CV fits**, so reproduction can be computationally expensive.
4. Retrieve `submission.csv` from notebook outputs and submit it to the competition.

For local execution, change the input paths and install packages from `requirements.txt`.

## Project Structure

```text
house-prices-prediction/
├── house-prices-prediction.ipynb
├── README.md
├── METHODOLOGY.md
├── RESULTS.md
├── requirements.txt
└── .gitignore
```

## Validation Limitations

The notebook performs target-informed feature screening and outlier removal **before** the train/holdout split, and evaluates its initial baseline on the holdout. Consequently, the holdout is **not strictly untouched** and the reported local metrics may be optimistic. Preprocessing inside the CV pipeline is fold-safe, but upstream feature selection is not. See [METHODOLOGY.md](METHODOLOGY.md) for details and recommended corrections.

## Tools

Python, pandas, NumPy, SciPy, matplotlib, seaborn, scikit-learn, XGBoost, Kaggle Notebooks.

## Dataset

[Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

The competition data are not included in this repository. Obtain them directly from Kaggle and follow the competition's data-use rules.
