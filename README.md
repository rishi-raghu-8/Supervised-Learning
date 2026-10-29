# Week 4: Supervised Learning Model Implementation

This project builds a binary classification model using Logistic Regression on the public Breast Cancer Wisconsin dataset available through scikit-learn.

## Contents
- `Week4_Supervised_Learning_Report.docx` — detailed report
- `supervised_learning.py` — complete Python implementation
- `breast_cancer_dataset.csv` — dataset with readable diagnosis labels
- `evaluation_metrics.csv` — recorded model metrics
- `test_set_predictions.csv` — held-out test predictions
- `visualizations/` — confusion matrix, ROC curve, coefficient chart, and cross-validation chart
- `requirements.txt` — Python dependencies

## Run locally
```powershell
pip install -r requirements.txt
python supervised_learning.py
```
The script writes generated plots and result files into `outputs/`.

## Method
- Stratified 80/20 train-test split
- StandardScaler + LogisticRegression pipeline
- Five-fold stratified cross-validation on training data
- Evaluation with accuracy, malignant-class precision/recall/F1, ROC-AUC, and confusion matrix

## Dataset
The dataset is provided by scikit-learn and originates from the Breast Cancer Wisconsin (Diagnostic) dataset. This is an educational demonstration, not a medical diagnostic tool. Clinical use requires extensive independent validation and qualified professional oversight.
