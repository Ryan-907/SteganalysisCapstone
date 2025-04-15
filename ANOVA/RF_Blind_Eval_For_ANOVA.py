import pandas as pd
import joblib

model = joblib.load("Metadata/models/random_forest_blind_v2.pkl")
df = pd.read_csv("Metadata/csv/blind_features_v2.csv")
X = df.drop(columns=["filename", "label"])
y = df["label"]
filenames = df["filename"]

results = []
y_pred = model.predict(X)
for true, pred, fname in zip(y, y_pred, filenames):
    binary_true = 0 if true == 0 else 1
    binary_pred = 0 if pred == 0 else 1
    correct = int(binary_true == binary_pred)
    results.append({"filename": fname, "model_type": "RF_Blind", "correct": correct})

pd.DataFrame(results).to_csv("Metadata/csv/RF_Blind_predictions_binary.csv", index=False)
print("✅ Saved RF Blind binary predictions to Metadata/csv/RF_Blind_predictions_binary.csv")
