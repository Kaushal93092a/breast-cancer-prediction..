"""
Use the trained model to predict on new patients.

Usage:
    python predict.py                      # predicts the bundled sample_patients.csv
    python predict.py my_patients.csv      # your own CSV (same 30 feature columns)
"""
import sys

import joblib
import pandas as pd

LABELS = {0: "Malignant", 1: "Benign"}

bundle = joblib.load("models/best_model.joblib")
model, features = bundle["model"], bundle["features"]

path = sys.argv[1] if len(sys.argv) > 1 else "sample_patients.csv"
df = pd.read_csv(path)
missing = [c for c in features if c not in df.columns]
if missing:
    sys.exit(f"Missing columns in {path}: {missing[:5]}...")

pred = model.predict(df[features])
conf = model.predict_proba(df[features]).max(axis=1)

for i, (p, c) in enumerate(zip(pred, conf), 1):
    line = f"Patient {i}: {LABELS[p]} (confidence {c:.1%})"
    if "actual" in df.columns:
        line += f"  | actual: {LABELS[int(df['actual'].iloc[i - 1])]}"
    print(line)
print("\nNote: educational project, not a medical diagnosis tool.")
