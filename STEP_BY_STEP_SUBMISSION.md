# Step-by-Step: Complete and Submit Your Final AI & ML Project

This follows the 3 key objectives of the module. Your project folder is already built and tested.

---

## ✅ Objective 1 – Build a complete AI / ML project

Already done for you. The project is **Breast Cancer Prediction** and includes data exploration, 5 compared models, tuning, evaluation, a saved model and a prediction script.

**Run it yourself once so you understand it (recommended):**

1. Install Python 3.9+ from https://python.org if you don't have it.
2. Open a terminal / Command Prompt inside the project folder.
3. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
4. Train the model (takes under a minute):
   ```
   python train.py
   ```
5. Try the predictions:
   ```
   python predict.py
   ```
6. Open the `outputs/` folder and look at the charts. Read `train.py` from top to bottom. You should be able to explain each of the 8 numbered sections.

---

## ✅ Objective 2 – Publish on GitHub with a proper README

### Option A – Upload through the website (easiest)

1. Go to https://github.com and sign in (or create a free account).
2. Click the **+** icon (top right) → **New repository**.
3. Fill in:
   - Repository name: `breast-cancer-prediction`
   - Description: `ML project predicting malignant vs benign tumours with scikit-learn`
   - Select **Public**
   - Leave "Add a README file" **unticked** (you already have one)
4. Click **Create repository**.
5. Click **uploading an existing file**.
6. Unzip the project, then drag these into the browser:
   - `train.py`, `predict.py`, `sample_patients.csv`
   - `requirements.txt`, `README.md`, `LICENSE`, `.gitignore`
   - the `outputs` folder (the PNG images and metrics.json)
   - the `models` folder (`best_model.joblib`)
   
   Tip: if GitHub does not accept folders by dragging, open the folder and select all the files inside it, then drag them. Or use Option B.
7. Scroll down, write the commit message `Initial commit: breast cancer prediction project`, and click **Commit changes**.
8. Open your repo's main page and check that the **README shows the charts and tables** correctly.

### Option B – Using Git commands

```
cd path/to/ml-final-project
git init
git add .
git commit -m "Initial commit: breast cancer prediction project"
git branch -M main
git remote add origin https://github.com/<your-username>/breast-cancer-prediction.git
git push -u origin main
```

### Fix the README link
In `README.md`, replace `<your-username>` with your real GitHub username (click the pencil icon on GitHub to edit the file).

---

## ✅ Objective 3 – Review your work and submit through the portal

**Review checklist (tick each one):**

- [ ] Repository is **Public** (open the link in a private/incognito window to confirm)
- [ ] README is visible and the images load
- [ ] README has your real username in the clone command
- [ ] `train.py`, `predict.py`, `requirements.txt` and `models/` are in the repo
- [ ] You can explain: what the dataset is, why you scaled the data, why Logistic Regression won, and what recall means

**LinkedIn Post (second deliverable):**

1. Open `LINKEDIN_POST.md` and copy the text.
2. Replace `<your-github-repo-link>` with your repo URL.
3. On LinkedIn click **Start a post**, paste the text.
4. Add images: upload `outputs/confusion_matrix.png` and `outputs/model_comparison.png` (or the cover image).
5. Click **Post**, then copy the link to your post (click the three dots on the post → **Copy link to post**).

**Submit:**

1. Open the course portal and go to the **Final AI & ML Project** module.
2. Click **Submit Deliverables**.
3. Paste your **GitHub repo link** and your **LinkedIn post link**.
4. Submit, then check that the status shows as submitted. This unlocks your milestone reward (the ATS resume templates).

---

## 🎤 If someone asks about your project (viva / interview)

- **What problem?** Classify tumours as malignant or benign from 30 cell measurements.
- **What data?** Breast Cancer Wisconsin, 569 samples.
- **Why scale features?** Logistic Regression, KNN and SVM are sensitive to feature ranges.
- **Why cross-validation?** It gives a more reliable estimate than a single split.
- **Why Logistic Regression won?** The classes are almost linearly separable, so a simple model generalises best.
- **Most important metric?** Recall for malignant cases, since a missed cancer is the costliest error.
- **Limitations?** Small dataset, single source, not a medical device.
