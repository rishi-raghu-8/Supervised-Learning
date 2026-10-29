"""
Week 4: Supervised Learning Model Implementation
Problem: Classify breast tumor observations as malignant or benign.
Dataset: Breast Cancer Wisconsin dataset included in scikit-learn.
Model: StandardScaler + LogisticRegression in a pipeline.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
)

OUT = Path("outputs")
PLOTS = OUT / "visualizations"
PLOTS.mkdir(parents=True, exist_ok=True)

dataset = load_breast_cancer(as_frame=True)
X = dataset.data.copy()
y = pd.Series(dataset.target, name="target")  # 0=malignant, 1=benign
labels = {0: "malignant", 1: "benign"}

# Stratified split preserves class proportions in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scaling is learned from training folds only because it is inside the Pipeline.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob_malignant = model.predict_proba(X_test)[:, 0]

print("Dataset shape:", X.shape)
print("Missing values:", int(X.isna().sum().sum()))
print("Test accuracy:", round(accuracy_score(y_test, pred), 4))
print("Malignant precision:", round(precision_score(y_test, pred, pos_label=0), 4))
print("Malignant recall:", round(recall_score(y_test, pred, pos_label=0), 4))
print("Malignant F1-score:", round(f1_score(y_test, pred, pos_label=0), 4))
print("Malignant ROC-AUC:", round(roc_auc_score((y_test == 0).astype(int), prob_malignant), 4))
print("5-fold CV accuracy mean:", round(cv_scores.mean(), 4))
print("5-fold CV accuracy std:", round(cv_scores.std(), 4))

# Confusion matrix
cm = confusion_matrix(y_test, pred, labels=[0, 1])
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm, display_labels=["Malignant", "Benign"]).plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix — Test Set")
fig.tight_layout()
fig.savefig(PLOTS / "01_confusion_matrix.png", dpi=180)
plt.close(fig)

# ROC curve, with malignant as the positive class
RocCurveDisplay.from_predictions((y_test == 0).astype(int), prob_malignant,
                                 name="Logistic Regression")
plt.title("ROC Curve — Malignant as Positive Class")
plt.tight_layout()
plt.savefig(PLOTS / "02_roc_curve.png", dpi=180)
plt.close()

# Coefficient magnitudes are a model-based association measure, not causal effects.
coef = pd.Series(abs(model.named_steps["classifier"].coef_[0]), index=X.columns).sort_values().tail(10)
fig, ax = plt.subplots()
coef.plot(kind="barh", ax=ax)
ax.set_xlabel("Absolute coefficient (standardized features)")
ax.set_title("Top 10 Features by Absolute Model Coefficient")
fig.tight_layout()
fig.savefig(PLOTS / "03_top_features.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots()
pd.Series(cv_scores, index=[f"Fold {i}" for i in range(1, 6)]).plot(kind="bar", ax=ax)
ax.axhline(cv_scores.mean(), linestyle="--", label=f"Mean = {cv_scores.mean():.3f}")
ax.set_ylabel("Accuracy")
ax.set_title("Five-Fold Stratified Cross-Validation Accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS / "04_cross_validation_accuracy.png", dpi=180)
plt.close(fig)

OUT.mkdir(exist_ok=True)
pd.DataFrame([{
    "test_accuracy": accuracy_score(y_test, pred),
    "malignant_precision": precision_score(y_test, pred, pos_label=0),
    "malignant_recall": recall_score(y_test, pred, pos_label=0),
    "malignant_f1": f1_score(y_test, pred, pos_label=0),
    "malignant_roc_auc": roc_auc_score((y_test == 0).astype(int), prob_malignant),
    "cv_accuracy_mean": cv_scores.mean(),
    "cv_accuracy_std": cv_scores.std()
}]).to_csv(OUT / "evaluation_metrics.csv", index=False)

predictions = X_test.copy()
predictions["actual_diagnosis"] = y_test.map(labels)
predictions["predicted_diagnosis"] = pd.Series(pred, index=y_test.index).map(labels)
predictions.to_csv(OUT / "test_set_predictions.csv", index=True, index_label="original_row_index")
print(f"Outputs saved in: {OUT.resolve()}")
