# Credit Card Fraud Detection using Machine Learning

A complete end-to-end fraud detection project built on the Kaggle Credit Card Fraud Detection dataset, including:

* Exploratory Data Analysis (EDA)
* Leakage-safe preprocessing
* Feature engineering
* Classical ML experimentation
* Threshold optimization
* Model evaluation on imbalanced data
* Production-style training pipeline

Dataset:
Kaggle Credit Card Fraud Detection Dataset [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

# Project Overview

This repository contains both:

1. **Research & experimentation notebook** (`fraud_nb.ipynb`)
2. **Production-oriented training pipeline** (`train.py` and modular src files)

The notebook covers the complete experimentation workflow:

* Exploratory data analysis
* Class imbalance analysis
* Fraud distribution across time
* Correlation analysis
* Feature engineering experiments
* Model experimentation
* Evaluation metric comparisons
* Threshold tuning
* Random Forest and XGBoost based approaches

The modular training pipeline converts the experimentation into reusable and maintainable production-style code.

---

# Dataset Description

The dataset contains anonymized European credit card transactions made in September 2013.

## Key Characteristics

| Property                | Value                   |
| ----------------------- | ----------------------- |
| Total Transactions      | 284,807                 |
| Fraudulent Transactions | 492                     |
| Fraud Rate              | ~0.172%                 |
| Problem Type            | Binary Classification   |
| Challenge               | Extreme Class Imbalance |

## Features

| Feature    | Description                                   |
| ---------- | --------------------------------------------- |
| `V1 - V28` | PCA-transformed anonymized numerical features |
| `Time`     | Seconds elapsed between transactions          |
| `Amount`   | Transaction amount                            |
| `Class`    | Target label (`1 = Fraud`, `0 = Legitimate`)  |

---

# Repository Structure

```bash
project/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud_nb.ipynb
│
├── models/
│   ├── rf_model.joblib
│   ├── scaler.joblib
│   └── metadata.json
│
├── results/
│
├── src/
│   ├── preprocessing.py
│   ├── metrics.py
│   └── train.py
│
├── config.py
├── requirements.txt
└── README.md
```

---

# Notebook Contents (`fraud_nb.ipynb`)

The notebook documents the full experimentation lifecycle.

## Exploratory Data Analysis

Includes:

* Fraud vs non-fraud distribution
* Fraud rate visualization
* Time-based fraud analysis
* Transaction amount distribution
* Correlation heatmaps
* Multicollinearity inspection

## Leakage-Safe Preprocessing

The preprocessing pipeline avoids data leakage by:

* Chronological splitting
* Fitting scalers only on training data
* Preserving temporal order

## Feature Engineering

Experiments include:

* Transaction amount scaling
* Time-derived features
* Feature matrix construction

## Modeling Experiments

The notebook explores multiple approaches including:

* Random Forest
* XGBoost
* Imbalance-aware learning strategies
* Threshold calibration
* PR-AUC optimization

## Evaluation

Focus is placed on fraud-sensitive metrics rather than raw accuracy.

Metrics used:

* PR-AUC
* ROC-AUC
* Precision
* Recall
* F1-score
* Confusion Matrix

---

# Production Training Pipeline

The `train.py` module provides a reusable training pipeline.

## Pipeline Steps

### 1. Data Loading

* Reads raw CSV
* Sorts transactions chronologically using `Time`

### 2. Time-Based Split

Chronological splitting prevents temporal leakage:

* Train set
* Validation set
* Test set

### 3. Preprocessing

* Scaling of transaction amount
* Optional time feature engineering
* Feature matrix generation

### 4. Hyperparameter Tuning

Random search tuning is performed using:

* `ParameterSampler`
* Validation PR-AUC as optimization criterion

### 5. Threshold Calibration

Instead of using the default probability threshold (`0.5`), the optimal threshold is selected based on:

* Maximum validation F1-score

### 6. Final Evaluation

The tuned model is evaluated exactly once on the held-out test set.

### 7. Artifact Persistence

The pipeline saves:

* Trained model
* Scaler
* Metadata JSON
* Evaluation metrics
* Best hyperparameters

---

# Model Details

## Primary Model

### Random Forest Classifier

Configuration highlights:

* `balanced_subsample` class weighting
* Random-search hyperparameter tuning
* Validation-based threshold optimization
* PR-AUC driven model selection

## Additional Experimentation

The notebook also includes experimentation with:

### XGBoost

Used for:

* Gradient boosting based fraud detection
* Performance benchmarking against Random Forest
* Imbalanced classification experimentation

---

# Why PR-AUC Instead of Accuracy?

This dataset is extremely imbalanced.

A model predicting all transactions as legitimate would still achieve very high accuracy.

Therefore, metrics such as:

* PR-AUC
* Recall
* Precision
* F1-score

are more meaningful for fraud detection.

---

# Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Notebook

```bash
jupyter notebook fraud_nb.ipynb
```

The notebook reproduces:

* EDA
* Feature engineering
* Model experimentation
* Evaluation
* Visualization

---

# Running the Training Pipeline

```bash
python src/train.py
```

The script will:

* Train the Random Forest model
* Tune hyperparameters
* Optimize threshold
* Evaluate on test data
* Save artifacts

---

# Saved Artifacts

| Artifact          | Description                                      |
| ----------------- | ------------------------------------------------ |
| `rf_model.joblib` | Trained Random Forest model                      |
| `scaler.joblib`   | Amount scaler                                    |
| `metadata.json`   | Metrics, threshold, feature columns, best params |

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib

---

# Future Improvements

Potential extensions:

* LightGBM implementation
* Deep learning anomaly detection
* Real-time inference API
* SHAP explainability
* Ensemble methods
* Streaming fraud detection
* Cost-sensitive learning

---

# References

* Kaggle Dataset Page [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* Dal Pozzolo et al., *Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy*

---

# License

This project is intended for educational, research, and portfolio purposes.

---

