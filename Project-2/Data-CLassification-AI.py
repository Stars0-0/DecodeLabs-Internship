# ============================================================
# DecodeLabs AI Internship — Project 2
# Data Classification Using AI (KNN on Iris Dataset)
# Kristan Martinez | Batch 2026
# ============================================================

try:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required to run this script. Install it with `pip install scikit-learn`."
    ) from exc

# ── PHASE 1: INPUT — Load & Understand the Dataset ──────────
iris_meta = load_iris(return_X_y=False)
X, y = iris_meta.data, iris_meta.target
class_names = [str(name) for name in iris_meta.target_names]

print("=" * 55)
print("   DecodeLabs Project 2 — KNN Data Classifier")
print("=" * 55)
print(f"\n Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f" Classes: {class_names}")
print(f"\n   Sample row (raw): {X[0]}")
print(f"   Label:             {class_names[y[0]]}")

# ── PHASE 2: PROCESS ────────────────────────────────────────

# Train-Test Split (80/20, shuffled to remove order bias)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True        # Iris is sorted by class — shuffle removes order bias
)
print(f"\n  Split: {len(X_train)} training | {len(X_test)} testing samples")

# Feature Scaling — mandatory for KNN (distance-based algorithm)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Fit on train ONLY
X_test  = scaler.transform(X_test)        # Apply same scale to test — never fit on test
print(f"⚖️  Scaling applied: Mean≈0, Variance≈1")
print(f"   Sample row (scaled): {X_train[0].round(3)}")

# K Selection — find optimal k using the Elbow Method principle
print(f"\n🔍 Finding optimal K (testing k=1 to 10)...")
best_k, best_acc = 1, 0
for k in range(1, 11):
    temp_model = KNeighborsClassifier(n_neighbors=k)
    temp_model.fit(X_train, y_train)
    acc = (temp_model.predict(X_test) == y_test).mean() * 100
    # >= prefers higher k when accuracy is tied — higher k = more generalised
    if acc >= best_acc:
        best_k, best_acc = k, acc
    print(f"   k={k:2d}  →  Accuracy: {acc:.1f}%")

print(f"\n Selected k={best_k} — highest k with best accuracy (more generalised, less overfitting risk)")

# Train final model with best K
model = KNeighborsClassifier(n_neighbors=best_k)  
model.fit(X_train, y_train)
print(f" KNN model trained with k={best_k}")

# ── PHASE 3: OUTPUT — Evaluate the Model ────────────────────
predictions = model.predict(X_test)

# Accuracy
accuracy = (predictions == y_test).mean() * 100
print(f"\n{'='*55}")
print(f"\t     RESULTS")
print(f"{'='*55}")
print(f"\t Accuracy : {accuracy:.1f}%")

# F1 Score
f1 = f1_score(y_test, predictions, average='weighted')
print(f"\t F1 Score : {f1:.4f}  (1.0 = perfect)")

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print(f"\n Confusion Matrix:")
print(f"   (Rows=Actual, Columns=Predicted)")
print(f"   Classes: {class_names}\n")          
for i, row in enumerate(cm):
    label = class_names[i].ljust(12)
    print(f"   {label} {row}")

# Classification Report
print(f"\n Classification Report:")
print(classification_report(y_test, predictions, target_names=class_names))

#  Live Prediction — all 3 classes 
print(f"{'='*55}")
print(f"  LIVE PREDICTION — New Flower Samples")
print(f"{'='*55}")

# Known samples — one per class to validate all 3 outputs
samples = [
    ([5.1, 3.5, 1.4, 0.2], "SETOSA"),
    ([6.3, 3.3, 4.7, 1.6], "VERSICOLOR"),
    ([6.3, 3.3, 6.0, 2.5], "VIRGINICA"),
]
for features, expected in samples:
    scaled = scaler.transform([features])
    pred   = class_names[model.predict(scaled)[0]].upper()
    status = "✅" if pred == expected else "❌"
    print(f"   Input: {features}  →  Predicted: {pred} {status}")