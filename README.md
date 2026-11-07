# 🩺 Breast Cancer Prediction – Final AI & ML Project

A complete end-to-end Machine Learning project that predicts whether a breast tumour is **malignant** or **benign** from 30 measurements of a cell nucleus. It covers data exploration, model comparison, hyperparameter tuning, evaluation, and a ready-to-use prediction script.

> ⚠️ **Educational project only.** This is not a medical diagnosis tool.

---

## 📌 Problem Statement

Early and accurate detection of malignant tumours is critical. Given measurements taken from a digitised image of a fine-needle aspirate of a breast mass, build a model that classifies the tumour as malignant or benign.

## 📊 Dataset

- **Breast Cancer Wisconsin (Diagnostic)** dataset, bundled with scikit-learn (no download needed)
- 569 samples, 30 numeric features (radius, texture, perimeter, area, smoothness, etc.)
- Classes: 212 malignant, 357 benign

## 🧠 Approach

1. **Explore** the data (class balance, feature correlations)
2. **Split** into 80% train / 20% test (stratified, fixed seed)
3. **Compare** 5 models with 5-fold cross-validation: Logistic Regression, KNN, SVM (RBF), Random Forest, Gradient Boosting
4. **Tune** the strongest candidates with `GridSearchCV`
5. **Evaluate** the final model once on the unseen test set
6. **Save** the model and use it through `predict.py`

All preprocessing (scaling) lives inside a scikit-learn `Pipeline`, so there is no data leakage between train and test.

## 🏆 Results

5-fold cross-validation on the training set:

| Model | CV accuracy |
|---|---|
| **Logistic Regression** | **97.80%** |
| SVM (RBF) | 96.92% |
| Random Forest | 96.48% |
| KNN | 96.26% |
| Gradient Boosting | 95.16% |

After tuning, **Logistic Regression (C = 0.1)** was selected (CV accuracy 98.24%).

**Final test-set performance (114 unseen patients):**

| Metric | Score |
|---|---|
| Accuracy | 97.37% |
| Precision (benign class) | 97.26% |
| Recall (benign class) | 98.61% |
| Recall (malignant class) | 95.24% |
| F1-score | 97.93% |
| ROC-AUC | 0.9957 |

The model missed 2 malignant cases and wrongly flagged 1 benign case out of 114 patients.

| Model comparison | Confusion matrix |
|---|---|
| ![comparison](outputs/model_comparison.png) | ![confusion](outputs/confusion_matrix.png) |

| ROC curve | Top features |
|---|---|
| ![roc](outputs/roc_curve.png) | ![features](outputs/feature_importance.png) |

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/breast-cancer-prediction.git
cd breast-cancer-prediction

# 2. (Optional) create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Mac/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train and evaluate (creates outputs/ and models/)
python train.py

# 5. Predict on sample patients (or pass your own CSV)
python predict.py
python predict.py my_patients.csv
```

Example output of `predict.py`:

```
Patient 1: Malignant (confidence 100.0%)  | actual: Malignant
Patient 2: Benign (confidence 99.9%)  | actual: Benign
```

## 📁 Project Structure

```
breast-cancer-prediction/
├── train.py              # EDA, model comparison, tuning, evaluation, saving
├── predict.py            # load saved model and predict on a CSV
├── sample_patients.csv   # 5 example patients
├── requirements.txt
├── models/
│   └── best_model.joblib # trained pipeline
├── outputs/              # charts + metrics.json
├── LICENSE
└── README.md
```

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib

## ⚠️ Limitations & Future Work

- Small dataset (569 samples), so results may not generalise to other hospitals or imaging devices
- In a medical setting, missing a malignant case (false negative) is costly, so a real system would tune the decision threshold for higher recall on malignant cases
- Next steps: add a Streamlit web app, try neural networks, explain predictions with SHAP

## 📄 License

MIT
