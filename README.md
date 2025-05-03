# COPPA Risk Classification

This project aims to build a machine learning pipeline to classify COPPA (Children's Online Privacy Protection Act) risk levels based on structured data. The notebook includes complete steps from data loading, exploratory data analysis (EDA), preprocessing, modeling, hyperparameter tuning, and model evaluation.

## Dataset

The dataset includes the following files, stored in Google Drive:

- `train.csv`: Training features
- `target.csv`: Target labels for the training set
- `test.csv`: Test set used for final predictions

## Dependencies

Key Python libraries used:

- Data manipulation: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`
- Statistics: `scipy`
- Modeling: `sklearn`, `catboost`, `xgboost`, `lightgbm`
- Hyperparameter tuning: `optuna`
- Imputation: `KNNImputer`
- Pipelines and preprocessing: `sklearn.preprocessing`, `sklearn.compose`

## Exploratory Data Analysis (EDA)

The EDA section covers:

- Missing values and duplicates
- Data type and unique value inspection
- Correlation analysis via heatmaps
- Class distribution and feature overview

## Preprocessing

Preprocessing steps include:

- Merging training data with target labels
- Handling missing values using KNN imputation
- One-hot encoding for categorical features
- Standardization for numerical features
- Feature selection and transformation pipelines

## Modeling

The following models are implemented:

- Random Forest
- CatBoost
- XGBoost
- LightGBM
- Logistic Regression
- Ensemble using VotingClassifier (hard and soft voting)

Models are evaluated using:

- Cross-validation (`StratifiedKFold`)
- ROC-AUC score
- Precision-Recall Curve
- Classification report (precision, recall, F1-score)

## Hyperparameter Tuning

Optuna is used to optimize hyperparameters for selected models, particularly CatBoost. The best parameters are selected based on validation metrics.

## Output

- Final models are selected based on validation performance
- Predictions are generated for the test set
- Results are saved for downstream submission or evaluation

## How to Run

1. Open the notebook in Google Colab
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
