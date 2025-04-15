import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

features_csv = "Metadata/csv/blind_features_v2.csv"
model_output = "Metadata/models/random_forest_blind_binary_consistent.pkl"
results_csv = "Metadata/csv/RF_Blind_predictions_binary_consistent.csv"
conf_matrix_img = "Metadata/images/ConfusionMatrixBlind_Binary_Consistent.png"

os.makedirs("Metadata/models", exist_ok=True)
os.makedirs("Metadata/images", exist_ok=True)
os.makedirs("Metadata/csv", exist_ok=True)

df = pd.read_csv(features_csv)
df["binary_label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

X = df.drop(columns=["filename", "label", "binary_label"])
y = df["binary_label"]
filenames = df["filename"]

X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
    X, y, filenames, test_size=0.25, stratify=y, random_state=42
)


print("🌲 Training Random Forest (Binary, Consistent Split)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
results = []

for true, pred, fname in zip(y_test, y_pred, f_test):
    results.append({
        "filename": fname,
        "model_type": "RF_Blind",
        "true_label": true,
        "predicted_label": pred,
        "correct": int(true == pred)
    })

df_results = pd.DataFrame(results)
df_results.to_csv(results_csv, index=False)
print(f"✅ Saved evaluation results to: {results_csv}")


accuracy = df_results["correct"].mean()
print(f"\n✅ Accuracy: {accuracy:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Clean", "Stego"]))


cm = confusion_matrix(y_test, y_pred)
print("\n=== Confusion Matrix ===")
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Clean", "Stego"], yticklabels=["Clean", "Stego"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Blind Binary - Consistent Split)")
plt.tight_layout()
plt.savefig(conf_matrix_img)
plt.close()
print(f"📊 Confusion matrix image saved to: {conf_matrix_img}")


joblib.dump(model, model_output)
print(f"💾 Model saved to: {model_output}")
