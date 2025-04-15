import os
import numpy as np
import pandas as pd
from skimage.io import imread
from scipy.stats import entropy, skew, kurtosis
from tqdm import tqdm

# === USER-DEFINED PATHS ===
image_dir = 'D:\CapstoneV2\DataSets\COCOTrainingImagespng2017'     
metadata_csv = 'D:\CapstoneV2\Metadata\csv\stego_metadata.csv'
output_csv = 'MetaData/csv/blind_features.csv'

# === Feature Extraction ===

def extract_features(image_gray):
    features = []

    # Bitplane features
    for bit in range(8):
        bitplane = ((image_gray >> bit) & 1).astype(np.uint8)
        mean_bp = np.mean(bitplane)
        var_bp = np.var(bitplane)
        energy_bp = np.sum(np.square(bitplane))
        counts = np.bincount(bitplane.flatten(), minlength=2)
        ent_bp = entropy(counts + 1e-8)
        features.extend([mean_bp, var_bp, energy_bp, ent_bp])

    # Global features
    image_flat = image_gray.flatten()
    mean_int = np.mean(image_flat)
    var_int = np.var(image_flat)
    counts = np.bincount(image_flat, minlength=256)
    ent_int = entropy(counts + 1e-8)
    skewness = skew(image_flat)
    kurt = kurtosis(image_flat)
    features.extend([mean_int, var_int, ent_int, skewness, kurt])

    return features

# === Load metadata ===

metadata = pd.read_csv(metadata_csv)

# === Extract Features ===

all_features = []

print("Extracting BLIND features...")

for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
    filename = row['Filename']
    label = row['Label']

    img_path = os.path.join(image_dir, filename)

    if not os.path.exists(img_path):
        print(f"Skipping {filename}: missing image.")
        continue

    image = imread(img_path, as_gray=True)

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    feats = extract_features(image)
    all_features.append([filename] + feats + [label])

# === Save CSV ===

bitplane_cols = [f'{stat}_bitplane{b}' for b in range(8) for stat in ['mean', 'var', 'energy', 'entropy']]
global_cols = ['mean_intensity', 'var_intensity', 'entropy_intensity', 'skewness', 'kurtosis']

columns = ['filename'] + bitplane_cols + global_cols + ['label']

df_out = pd.DataFrame(all_features, columns=columns)
df_out.to_csv(output_csv, index=False)
print(f"Saved blind features to {output_csv}")
