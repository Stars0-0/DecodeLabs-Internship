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
| 🤖 KNN Classifier | K=5 neighbors, majority vote classification |
| 🔢 Confusion Matrix | Full TP/FP/FN/TN breakdown per class |
| 🎯 F1 Score | Weighted harmonic mean of Precision & Recall |
| 🌸 Live Prediction | Classifies a brand-new unseen flower sample |

---

## 🏗️ Architecture

The project follows the **IPO Framework** introduced in the DecodeLabs briefing:

```
INPUT                    PROCESS                   OUTPUT
─────────────────────────────────────────────────────────
Iris Dataset         →   Train-Test Split (80/20)  →  Confusion Matrix
Feature Scaling          KNN Algorithm (k=5)          F1 Score
                         model.fit(X_train)            Live Prediction
                         model.predict(X_test)
```

**Why StandardScaler is mandatory for KNN:**
```
Raw data:    sepal_length ≈ 5.0   petal_length ≈ 1.4   ← similar scale
             BUT in other datasets features can range 0-1 vs 0-1000
             → larger feature dominates distance calculation → biased model

Scaled data: all features → Mean=0, Variance=1
             → every feature contributes equally to distance
```

---

## 🚀 How to Run

**Requirements:** Python 3.x + scikit-learn

# 1. Navigate to your project folder first
cd ~/Documents/GitHub/DecodeLabs-Internship/Project-2/Data-CLassification-AI

# 2. Create the virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate

# 4. Install scikit-learn
pip install scikit-learn

# 5. Run your project
python3 Data-CLassification-AI.py

**Expected Output:**
```
=======================================================
   DecodeLabs Project 2 — KNN Data Classifier
=======================================================

📦 Dataset loaded: 150 samples, 4 features
🌸 Classes: ['setosa', 'versicolor', 'virginica']

✂️  Split: 120 training | 30 testing samples
⚖️  Scaling applied: Mean≈0, Variance≈1

🤖 KNN model trained (k=5)

=======================================================
📊 RESULTS
=======================================================
✅ Accuracy:  100.0%
🎯 F1 Score:  1.0000  (1.0 = perfect)

🔢 Confusion Matrix:
   (Rows=Actual, Cols=Predicted)
   Classes: ['setosa', 'versicolor', 'virginica']

   setosa       [10  0  0]
   versicolor   [ 0  9  0]
   virginica    [ 0  0 11]

🌸 LIVE PREDICTION — New Flower Sample
Input features: sepal=5.1x3.5cm, petal=1.4x0.2cm
Predicted class: 👉 SETOSA
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
Project 2/
│
├── Data-Classification-AI.py    
└── README.md       
```

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | ~96–100% |
| F1 Score (weighted) | ~0.96–1.00 |
| Algorithm | KNeighborsClassifier (k=5) |
| Dataset | Iris (sklearn built-in) |

---



## 👤 Author

**Kristan Martinez**
AI Engineering Intern — DecodeLabs Batch 2026

---
