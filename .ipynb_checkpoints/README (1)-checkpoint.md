# TalentGuard-AI

A supervised machine learning project that predicts employee turnover using Logistic Regression.

## Overview

This project trains and evaluates logistic regression models on an employee dataset to classify whether an employee is likely to stay or leave the company. It explores the effect of L1 and L2 regularization and compares model performance across different hyperparameter settings.

## Dataset

- **File:** `employee_turnover.csv`
- **Rows:** 1,350 employees
- **Target:** `Employee_Turnover` (0 = Stayed, 1 = Left)
- **Features (15):** Job satisfaction, performance rating, age, monthly income, work-life balance, training hours, and more
- **Class balance:** Nearly equal — 678 stayed, 672 left

## What the Notebook Does

1. Loads and explores the dataset
2. Checks for missing values and class balance
3. Splits data into 80% train / 20% test
4. Scales features using `StandardScaler`
5. Trains three models:
   - **Baseline** — default Logistic Regression (C=1.0)
   - **L2 Ridge** — tested C values: 0.001 → 100
   - **L1 Lasso** — tested C values: 0.001 → 100
6. Evaluates each model with confusion matrix, accuracy, precision, recall, and F1-score

## Results

| Model | Best C | Accuracy |
|-------|--------|----------|
| Baseline | 1.0 | 86.30% |
| L2 Ridge | 1.0 | 86.30% |
| L1 Lasso | 0.1 | 86.30% |

**Key finding:** L1 regularization zeroed out 8 of 15 features while matching the same accuracy — indicating those features contributed little predictive value.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- Matplotlib

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook TalentGuard_AI.ipynb
```

---

