import pandas as pd
import joblib

model = joblib.load("Metadata/Models/random_forest_model.pkl")
df = pd.read_csv("Metadata/csv/diff_features.csv")
X = df.drop(columns=["filename", "label"])
y = df["label"]
filenames = df["filename"]

results = []
y_pred = model.predict(X)
for true, pred, fname in zip(y, y_pred, filenames):
    binary_true = 0 if true == 0 else 1
    binary_pred = 0 if pred == 0 else 1
    correct = int(binary_true == binary_pred)
    results.append({"filename": fname, "model_type": "RF_Paired", "correct": correct})

pd.DataFrame(results).to_csv("Metadata/csv/RF_Paired_predictions_binary.csv", index=False)
print("Saved RF Paired binary predictions to Metadata/csv/RF_Paired_predictions_binary.csv")
