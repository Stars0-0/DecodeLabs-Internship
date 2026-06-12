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


#Phase 1: Load and Preprocess the Data

iris_meta = load_iris(return_X_y=False)
x, y = iris_meta.data, iris_meta.target

print("=" * 55)
print("   DecodeLabs Project 2 — KNN Data Classifier")
print("=" * 55)
print(f"\n Dataset loaded: {x.shape[0]} samples, {x.shape[1]} features")
print(f" Classes: {list(iris_meta.target_names)}")
print(f"\nSample row (raw): {x[0]}")
print(f"Label:            {iris_meta.target_names[y[0]]}")


#Phase 2 : Process the Data

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

print(f"\n Split: {len(x_train)} training | {len(x_test)} testing samples")

scaler = StandardScaler()
X_train =  scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

print(f"\n Scaling applied: Mean=0, Variance = 1")
print(f" Sample row (scaled): {X_train[0].round(3)}")


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print(f"\n KNN model trained with k=5")

#Phase 3: Evaluate the Model

predictions = model.predict(X_test)

#Accuacy
accuracy = (predictions == y_test).mean()*100

print(f"\n{'='*55}")
print(f"\t RESULTS")
print(f"{'='*55}")
print(f"\t Accuracy:  {accuracy:.1f}%")

f1 = f1_score(y_test, predictions, average='weighted')
print(f"\t F1 Score:  {f1:.4f} (1.0 = perfect)")

#Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print(f"\n Confusion Matrix:")
print(f"  (Rows=Actual, Columns=Predicted)")
print(f"  Classses: {list(iris_meta.target_names)}\n")

for i, row in enumerate(cm):
    label = iris_meta.target_names[i].ljust(12)
    print(f"  {label} {row}")
    
#Classification Report
print(f"\n Classification Report:")
print(classification_report(y_test, predictions, target_names=iris_meta.target_names))

#Prediction Example

print(f"{'='*55}")
print(f" LIVE PREDICTION — New Flower Sample")
print(f"{'='*55}")
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Known Setosa measurements
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)
print(f"Input features: sepal=5.1x3.5cm, petal=1.4x0.2cm")
print(f"Predicted class: {iris_meta.target_names[prediction[0]].upper()}")