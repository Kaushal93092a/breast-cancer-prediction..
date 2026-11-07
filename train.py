"""
Breast Cancer Prediction - Final AI & ML Project
Trains, compares, tunes and evaluates several ML models, then saves the best one.

Run:  python train.py
"""
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)  # hide sklearn deprecation notices

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. Load data
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
X, y = data.data, data.target          # target: 0 = malignant, 1 = benign
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print("Class balance:\n", y.map({0: "malignant", 1: "benign"}).value_counts(), "\n")

# ---------------------------------------------------------------- 2. Quick EDA
fig, ax = plt.subplots(figsize=(5, 4))
y.map({0: "Malignant", 1: "Benign"}).value_counts().plot(kind="bar", color=["#4f46e5", "#ef4444"], ax=ax)
ax.set_title("Class distribution")
ax.set_ylabel("Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "class_distribution.png", dpi=150)
plt.close()

top_corr = df.corr()["target"].drop("target").abs().sort_values(ascending=False).head(10).index
plt.figure(figsize=(8, 6.5))
sns.heatmap(df[list(top_corr) + ["target"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlation of top 10 features with the target")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# ---------------------------------------------------------------- 4. Compare models
models = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
    "KNN": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
    "SVM (RBF)": Pipeline([("scale", StandardScaler()), ("clf", SVC(probability=True, random_state=SEED))]),
    "Random Forest": Pipeline([("clf", RandomForestClassifier(n_estimators=300, random_state=SEED))]),
    "Gradient Boosting": Pipeline([("clf", GradientBoostingClassifier(random_state=SEED))]),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}
print("5-fold cross-validation (training set):")
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = scores
    print(f"  {name:<20} {scores.mean():.4f} +/- {scores.std():.4f}")

plt.figure(figsize=(8, 4.5))
plt.boxplot(cv_results.values(), tick_labels=cv_results.keys())
plt.xticks(rotation=20)
plt.ylabel("CV accuracy")
plt.title("Model comparison (5-fold cross-validation)")
plt.tight_layout()
plt.savefig(OUT / "model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 5. Tune best candidates
grids = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "SVM (RBF)": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]},
}
tuned = {}
for name, grid in grids.items():
    gs = GridSearchCV(models[name], grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned[name] = gs
    print(f"Tuned {name}: best CV={gs.best_score_:.4f}  params={gs.best_params_}")

best_name = max(tuned, key=lambda n: tuned[n].best_score_)
best_model = tuned[best_name].best_estimator_
print(f"\nSelected model: {best_name}")

# ---------------------------------------------------------------- 6. Final evaluation on unseen test set
pred = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, 1]
metrics = {
    "model": best_name,
    "best_params": tuned[best_name].best_params_,
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "roc_auc": round(roc_auc_score(y_test, proba), 4),
    "cv_accuracy": round(tuned[best_name].best_score_, 4),
    "train_size": len(X_train),
    "test_size": len(X_test),
}
print("\nTest-set results:")
print(classification_report(y_test, pred, target_names=["malignant", "benign"]))
print(json.dumps(metrics, indent=2))
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Malignant", "Benign"], cmap="Purples", ax=ax)
ax.set_title(f"Confusion matrix - {best_name}")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=best_name)
ax.plot([0, 1], [0, 1], "k--", alpha=.4)
ax.set_title("ROC curve")
plt.tight_layout()
plt.savefig(OUT / "roc_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 7. Feature importance (Random Forest, for insight)
rf = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(X_train, y_train)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10)
plt.figure(figsize=(7, 4.5))
imp.plot(kind="barh", color="#4f46e5")
plt.title("Top 10 most important features (Random Forest)")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 8. Save model + a few demo samples
joblib.dump({"model": best_model, "features": list(X.columns)}, "models/best_model.joblib")
X_test.head(5).assign(actual=y_test.head(5)).to_csv("sample_patients.csv", index=False)
print("\nSaved: models/best_model.joblib, outputs/*.png, outputs/metrics.json, sample_patients.csv")
