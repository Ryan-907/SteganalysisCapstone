import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# === User-defined paths ===
features_csv = 'D:/CapstoneV2/Metadata/csv/blind_features_v2.csv'
model_output_path = 'Metadata/models/random_forest_blind_binary.pkl'
conf_matrix_output_path = 'Metadata/images/ConfusionMatrixBlind_Binary.png'

# === Ensure Metadata directories exist ===
os.makedirs('Metadata/models', exist_ok=True)
os.makedirs('Metadata/images', exist_ok=True)

# === Load Extracted Features ===
df = pd.read_csv(features_csv)

# === Binary Label Mapping: 0 = Clean, 1 = Stego (LSB or DCT) ===
df['binary_label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)

# === Separate features and labels ===
X = df.drop(columns=['filename', 'label', 'binary_label'])
y = df['binary_label']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# === Train Random Forest ===
print("Training Random Forest (Binary Classification)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)

print("\n=== Classification Report (Binary) ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix (Binary) ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# === Save Confusion Matrix Plot ===
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Clean', 'Stego'], yticklabels=['Clean', 'Stego'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Blind - Binary)")
plt.tight_layout()
plt.savefig(conf_matrix_output_path)
plt.close()

print(f"Confusion matrix saved to {conf_matrix_output_path}")

# === Save Model ===
joblib.dump(model, model_output_path)
print(f"Model saved to {model_output_path}")

# === Feature Importance ===
importances = model.feature_importances_
feature_names = X.columns
sorted_idx = importances.argsort()[::-1]

print("\n=== Top 15 Feature Importances ===")
for i in range(15):
    print(f"{feature_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")
