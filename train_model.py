import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Check if necessary columns exist
if 'Disease' not in df.columns:
    raise ValueError("Dataset must have a 'Disease' column.")
if 'Symptoms' not in df.columns:
    # Combine symptom columns if separate
    symptom_cols = [col for col in df.columns if 'Symptom' in col]
    df['Symptoms'] = df[symptom_cols].apply(
        lambda x: ','.join([s.strip().lower() for s in x.dropna().astype(str).tolist() if s.strip() != '']),
        axis=1
    )

# Convert symptom text into lists
df['Symptoms'] = df['Symptoms'].apply(lambda x: [s.strip().lower() for s in x.split(',') if s.strip() != ''])

# Binarize symptom features
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['Symptoms'])
y = df['Disease']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

# Slight noise to avoid overfitting
noise_factor = 0.08
X_train_noisy = X_train.copy()
mask = np.random.rand(*X_train_noisy.shape) < noise_factor
X_train_noisy[mask] = 1 - X_train_noisy[mask]

# Model 1: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=120,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train_noisy, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# Model 2: Logistic Regression
lr_model = LogisticRegression(max_iter=200, solver='liblinear')
lr_model.fit(X_train_noisy, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Model 3: Support Vector Classifier
svc_model = SVC(kernel='linear', probability=True)
svc_model.fit(X_train_noisy, y_train)
svc_pred = svc_model.predict(X_test)
svc_acc = accuracy_score(y_test, svc_pred)

# Print comparison
print("\nModel Comparison Results:")
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
print(f"SVC Accuracy: {svc_acc * 100:.2f}%")

# Pick best model automatically
models = {
    "RandomForest": (rf_model, rf_acc),
    "LogisticRegression": (lr_model, lr_acc),
    "SVC": (svc_model, svc_acc)
}
best_name, (best_model, best_acc) = max(models.items(), key=lambda x: x[1][1])

print(f"\nBest Model: {best_name} with accuracy {best_acc * 100:.2f}%")

# Save best model and encoder
pickle.dump(best_model, open("disease_model.pkl", "wb"))
pickle.dump(mlb, open("symptom_binarizer.pkl", "wb"))
print("\nModel and encoder saved: disease_model.pkl & symptom_binarizer.pkl")

# Classification report for best model
best_pred = best_model.predict(X_test)
print("\nClassification Report for Best Model:\n")
print(classification_report(y_test, best_pred))
