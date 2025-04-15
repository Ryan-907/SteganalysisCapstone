import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


csv_file = "Metadata\stego_metadata.csv"
image_dir = "D:\CapstoneV2\DataSets\COCOTrainingImagespng2017"
sample_size = 20


df = pd.read_csv(csv_file)
if 'Label' not in df.columns:
    raise ValueError("CSV must contain a 'Label' column.")

print("\n📊 Label Distribution:")
print(df['Label'].value_counts())

label_map = {0: 'Clean', 1: 'LSB', 2: 'DCT'}
class_samples = {0: [], 1: [], 2: []}

for _, row in df.iterrows():
    label = row['Label']
    if len(class_samples[label]) < 3:
        path = os.path.join(image_dir, row['Filename'])
        if os.path.exists(path):
            class_samples[label].append((path, label_map[label]))

fig, axs = plt.subplots(3, 3, figsize=(9, 9))
for i, label in enumerate([0, 1, 2]):
    for j, (path, lbl) in enumerate(class_samples[label]):
        img = Image.open(path).convert("RGB")
        axs[i, j].imshow(img)
        axs[i, j].set_title(lbl)
        axs[i, j].axis('off')
plt.suptitle("🔍 Sample Images by Class")
plt.tight_layout()
plt.show()

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

class SimpleDataset(torch.utils.data.Dataset):
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

small_df = df.sample(sample_size, random_state=42)
small_dataset = SimpleDataset(small_df.to_dict(orient='records'), image_dir, transform)
loader = DataLoader(small_dataset, batch_size=4, shuffle=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n🧪 Training on small sample to test overfitting...")
model.train()
for epoch in range(10):
    total_loss = 0
    correct, total = 0, 0
    for images, labels in loader:
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
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {correct}/{total} ({correct/total:.2%})")


print("\n📊 Generating confusion matrix for small sample:")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clean", "LSB", "DCT"])
disp.plot(cmap='Blues')
plt.title("🧮 Confusion Matrix (Overfit Sample)")
plt.show()
