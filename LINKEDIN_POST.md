🚀 Final AI & ML Project: complete!

I just finished my final project: a Machine Learning model that predicts whether a breast tumour is malignant or benign, using the Breast Cancer Wisconsin dataset. 🩺

Here's what I did:
✅ Explored the data (class balance + feature correlations)
✅ Compared 5 models with 5-fold cross-validation: Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting
✅ Tuned the best candidates with GridSearchCV
✅ Evaluated on 114 unseen patients

📊 Results:
• Accuracy: 97.4%
• Recall: 98.6% (benign), 95.2% (malignant)
• ROC-AUC: 0.996

💡 Key lessons:
• The simplest model (Logistic Regression) beat the complex ones, so always compare against a simple baseline
• Putting scaling inside a Pipeline prevents data leakage
• In healthcare, accuracy isn't everything: missing a malignant case is far more costly than a false alarm, so recall and the confusion matrix matter
• A project is only useful when others can run it, so I added a clear README and a predict script

(This is an educational project, not a diagnosis tool.)

🔗 GitHub: <your-github-repo-link>

Next up: a Streamlit web app and model explainability with SHAP.

Feedback is welcome! 🙌

#MachineLearning #AI #DataScience #Python #ScikitLearn #GitHub #LearningInPublic #FinalProject
