import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


features_csv = 'D:\CapstoneV2\Metadata\csv\diff_features.csv'
model_output_path = 'MetaData/models/random_forest_model.pkl'
conf_matrix_output_path = 'MetaData/images/CleanConfusionMmatrix.png'


df = pd.read_csv(features_csv)


X = df.drop(columns=['filename', 'label'])
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)


plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Clean', 'LSB', 'DCT'], yticklabels=['Clean', 'LSB', 'DCT'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(conf_matrix_output_path)
plt.close()  

print(f"Confusion matrix saved to {conf_matrix_output_path}")




os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

joblib.dump(model, model_output_path)
print(f"Model saved to {model_output_path}")


importances = model.feature_importances_
feature_names = X.columns
sorted_idx = importances.argsort()[::-1]

print("\n=== Top 15 Feature Importances ===")
for i in range(15):
    print(f"{feature_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")
