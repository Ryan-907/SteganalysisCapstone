import os
import pandas as pd
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image

csv_file = "Metadata\csv\stego_metadata.csv"
image_dir = "DataSets/COCOTrainingImagespng2017"
model_path = "Metadata/models/model_resnet18_stego.pt"
output_csv = "Metadata/csv/CNN_predictions_binary.csv"

class SimpleDataset(Dataset):
    def __init__(self, rows, image_dir, transform):
        self.rows = rows
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = os.path.join(image_dir, row['Filename'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(row['Label'])
        filename = row['Filename']
        return image, label, filename

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

df = pd.read_csv(csv_file)
dataset = SimpleDataset(df.to_dict(orient='records'), image_dir, transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

results = []
with torch.no_grad():
    for images, labels, filenames in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for true, pred, fname in zip(labels, preds.cpu(), filenames):
            binary_true = 0 if true == 0 else 1
            binary_pred = 0 if pred == 0 else 1
            correct = int(binary_true == binary_pred)
            results.append({"filename": fname, "model_type": "CNN", "correct": correct})

pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"Saved CNN binary predictions to {output_csv}")
