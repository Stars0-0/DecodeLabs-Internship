# 📊 Project 2: Data Classification Using AI

> **DecodeLabs Industrial Training | Batch 2026 | AI Engineering Track**

---

## 📌 Overview

A supervised machine learning classification pipeline built in Python using scikit-learn. This project demonstrates the complete **IPO Framework** for predictive AI — from raw data ingestion and feature scaling, through model training with the K-Nearest Neighbors algorithm, to validated output via Confusion Matrix and F1 Score.

This is the bridge from rule-based deterministic logic (Project 1) to data-driven probabilistic decision making.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📦 Dataset Loading | Iris benchmark dataset — 150 samples, 4 features, 3 classes |
| ⚖️ Feature Scaling | StandardScaler (Mean=0, Variance=1) — prevents distance bias in KNN |
| ✂️ Train-Test Split | 80/20 split with shuffle — removes order bias |
| 🤖 KNN Classifier | Auto-selected optimal K (k=1 to 10) via accuracy comparison |
| 🔢 Confusion Matrix | Full TP/FP/FN/TN breakdown per class |
| 🎯 F1 Score | Weighted harmonic mean of Precision & Recall |
| 🌸 Live Prediction | Classifies 3 brand-new unseen flower samples (one per class) |

---

## 🏗️ Architecture

The project follows the **IPO Framework** introduced in the DecodeLabs briefing:

```
INPUT                    PROCESS                        OUTPUT
──────────────────────────────────────────────────────────────
Iris Dataset         →   Train-Test Split (80/20)   →  Confusion Matrix
Feature Scaling          K Selection (k=1 to 10)       F1 Score
                         KNN Algorithm (best k)         Live Prediction
                         model.fit(X_train)
                         model.predict(X_test)
```

**Why StandardScaler is mandatory for KNN:**
```
Raw data:    features can range 0-1 vs 0-1000 in real datasets
             → larger feature dominates distance calculation → biased model

Scaled data: all features → Mean=0, Variance=1
             → every feature contributes equally to distance
```

---

## 🚀 How to Run

**Requirements:** Python 3.x

```bash
# 1. Navigate to the project folder
cd Project-2

# 2. Create and activate a virtual environment
python3 -m venv venv

#Linux
source venv/bin/activate

#Windows
venv\Scripts\activate

# 3. Install dependency
pip install scikit-learn

# 4. Run the classifier
python3 Data-CLassification-AI.py
```

**Expected Output:**
```
=======================================================
   DecodeLabs Project 2 — KNN Data Classifier
=======================================================
 Dataset loaded: 150 samples, 4 features
 Classes: ['setosa', 'versicolor', 'virginica']
Sample row (raw): [5.1 3.5 1.4 0.2]
Label:            setosa

 Split: 120 training | 30 testing samples
 Scaling applied: Mean=0, Variance = 1
 Sample row (scaled): [-1.474  1.204 -1.563 -1.313]

🔍 Finding optimal K (testing k=1 to 10)...
   k= 1  →  Accuracy: 100.0%
   k= 2  →  Accuracy: 100.0%
   ...
   k=10  →  Accuracy: 100.0%
 Selected k=10 — highest k with best accuracy (more generalised, less overfitting risk)
 KNN model trained with k=10

=======================================================
         RESULTS
=======================================================
         Accuracy:  100.0%
         F1 Score:  1.0000 (1.0 = perfect)

 Confusion Matrix:
  (Rows=Actual, Columns=Predicted)
  Classes: ['setosa', 'versicolor', 'virginica']

  setosa       [10  0  0]
  versicolor   [ 0  9  0]
  virginica    [ 0  0 11]

 Classification Report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11
    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

=======================================================
 LIVE PREDICTION — New Flower Sample
=======================================================
   Input: [5.1, 3.5, 1.4, 0.2]  →  Predicted: SETOSA ✅
   Input: [6.3, 3.3, 4.7, 1.6]  →  Predicted: VERSICOLOR ✅
   Input: [6.3, 3.3, 6.0, 2.5]  →  Predicted: VIRGINICA ✅
```

---

## 🧠 Concepts Demonstrated

- **Supervised Learning** — machine derives logic from labeled historical data
- **Train-Test Split** — structural integrity; test set locked until evaluation
- **Feature Scaling** — the "Gatekeeper Rule"; normalizes distance-sensitive algorithms
- **KNN Algorithm** — proximity principle: similar things exist in close proximity
- **The Elbow Method** — choosing optimal K to avoid overfitting (K=1) or underfitting (K=100)
- **Confusion Matrix** — diagnostic tool: TP, FP (Type I), FN (Type II), TN
- **F1 Score** — harmonic mean of Precision & Recall; superior to raw accuracy on imbalanced data
- **Accuracy Mirage** — why 99% accuracy can still be a failing model

---

## 📁 Project Structure

```
Project-2/
├── Data-CLassification-AI.py
├── README.md
└── .gitignore
```

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | 100.0% |
| F1 Score (weighted) | 1.0000 |
| Optimal K selected | k=10 |
| Algorithm | KNeighborsClassifier |
| Dataset | Iris (sklearn built-in) |

> **Note:** 100% accuracy is expected on the Iris dataset — it is a clean,
> balanced benchmark designed for algorithm validation. The key skill
> demonstrated is the pipeline architecture, not the score itself.

---

## 👤 Author

**Kristan Martinez**
AI Engineering Intern — DecodeLabs Batch 2026
🔗 [LinkedIn](#) | 🐙 [GitHub](#)

---
