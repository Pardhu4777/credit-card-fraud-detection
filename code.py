import pandas as pd
import numpy as np
from sdv.single_table.ctgan import CTGAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ----------------------------
# 1. Load Dataset
# ----------------------------
data = pd.read_csv("creditcard.csv", on_bad_lines='skip')

print("Dataset Shape:", data.shape)

# ----------------------------
# 2. Reduce Dataset Size (Optional but Recommended)
# ----------------------------
# Use smaller subset for faster Colab training
data = data.sample(n=50000, random_state=42)   # Reduce to 50k rows

# ----------------------------
# 3. Train-Test Split
# ----------------------------
X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# 4. Separate Fraud Cases
# ----------------------------
fraud_data = X_train[y_train == 1].copy()

print("Original Fraud Samples:", fraud_data.shape[0])

# Reduce fraud samples for CTGAN training (faster)
fraud_data = fraud_data.sample(
    n=min(200, len(fraud_data)), random_state=42
)

# ----------------------------
# 5. Train CTGAN (Reduced Epochs)
# ----------------------------
ctgan = CTGAN(epochs=50)   # Reduced from 300 → 50
ctgan.fit(fraud_data)

# ----------------------------
# 6. Generate Synthetic Fraud Samples (Reduced)
# ----------------------------
synthetic_fraud = ctgan.sample(1000)  # Reduced from 5000 → 1000

print("Synthetic Fraud Samples:", synthetic_fraud.shape[0])

# ----------------------------
# 7. Combine Real + Synthetic Data
# ----------------------------
X_augmented = pd.concat([X_train, synthetic_fraud], axis=0)
y_augmented = pd.concat([y_train, pd.Series([1]*1000)], axis=0)

# ----------------------------
# 8. Train Faster Random Forest
# ----------------------------
model = RandomForestClassifier(
    n_estimators=100,      # Reduced from 300 → 100
    max_depth=20,          # Limit depth
    n_jobs=-1,             # Use all CPU cores
    random_state=42
)

model.fit(X_augmented, y_augmented)

# ----------------------------
# 9. Evaluation
# ----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("\nClassification Report:")
# Convert classification report to a DataFrame for better display
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
display(report_df)

print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Create a heatmap of the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()
