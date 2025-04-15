import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

# === CONFIGURATION ===
csv_file = "Metadata/stego_metadata.csv"
image_dir = "D:/CapstoneV2/DataSets/COCOTrainingImagespng2017"
batch_size = 32
epochs = 10
test_size = 0.2
save_path = "model_resnet18_stego.pt"
log_csv = "training_log.csv"

# === Load Metadata ===
df = pd.read_csv(csv_file)
if 'Label' not in df.columns or 'Filename' not in df.columns:
    raise ValueError("CSV must contain 'Label' and 'Filename' columns.")
print("\n📊 Label Distribution:")
print(df['Label'].value_counts())

# === Dataset Class ===
class SimpleDataset(Dataset):
    def __init__(self, rows, image_dir, transform):
        self.rows = rows
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = os.path.join(self.image_dir, row['Filename'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(row['Label'])
        return image, label

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Train/Test Split ===
train_rows, test_rows = train_test_split(
    df.to_dict(orient='records'),
    test_size=test_size,
    stratify=df['Label'],
    random_state=42
)

train_loader = DataLoader(SimpleDataset(train_rows, image_dir, transform), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(SimpleDataset(test_rows, image_dir, transform), batch_size=batch_size)

# === Model Setup ===
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Training ===
print("\n🚀 Training full model...")
training_log = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct, total = 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=total_loss, acc=f"{correct/total:.2%}")

    acc = correct / total
    training_log.append({'epoch': epoch+1, 'loss': total_loss, 'accuracy': acc})

# === Save Model ===
torch.save(model.state_dict(), save_path)
print(f"\n💾 Model saved to: {save_path}")

# === Evaluation ===
print("\n🔍 Evaluating on test set...")
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

# === Classification Report ===
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Clean", "LSB", "DCT"]))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clean", "LSB", "DCT"])
disp.plot(cmap='Blues')
plt.title("📊 Confusion Matrix (Test Set)")
plt.show()

# === Optional: Save training log to CSV ===
# pd.DataFrame(training_log).to_csv(log_csv, index=False)
# print(f"📁 Training log saved to: {log_csv}")
