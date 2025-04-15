import os
import numpy as np
import pandas as pd
from skimage.io import imread
from scipy.stats import entropy, skew, kurtosis
from tqdm import tqdm
from scipy.fftpack import dct
from scipy.ndimage import gaussian_filter

# === PATHS ===
image_dir = 'D:/CapstoneV2/DataSets/COCOTrainingImagespng2017'
metadata_csv = 'D:/CapstoneV2/Metadata/csv/stego_metadata.csv'
output_csv = 'MetaData/csv/blind_features_v2.csv'

# === Feature Functions ===

def extract_dct_features(image_gray):
    h, w = image_gray.shape
    dct_feats = []
    for i in range(0, h - 8 + 1, 8):
        for j in range(0, w - 8 + 1, 8):
            block = image_gray[i:i+8, j:j+8]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_feats.extend(dct_block.flatten())
    dct_feats = np.array(dct_feats)
    return [
        np.mean(dct_feats),
        np.var(dct_feats),
        entropy(np.histogram(dct_feats, bins=64, range=(0, 255))[0] + 1e-8)
    ]

def extract_residual_features(image_gray):
    blurred = gaussian_filter(image_gray, sigma=1)
    residual = image_gray - blurred
    flat = residual.flatten()
    return [
        np.mean(flat),
        np.var(flat),
        entropy(np.histogram(flat, bins=64)[0] + 1e-8)
    ]

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
    flat = image_gray.flatten()
    features.extend([
        np.mean(flat),
        np.var(flat),
        entropy(np.bincount(flat, minlength=256) + 1e-8),
        skew(flat),
        kurtosis(flat)
    ])

    # Residual features
    features.extend(extract_residual_features(image_gray))

    # DCT features
    features.extend(extract_dct_features(image_gray))

    return features

# === Load Metadata ===
metadata = pd.read_csv(metadata_csv)
all_features = []

print("🔍 Extracting enhanced blind features...")

for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
    filename = row['Filename']
    label = row['Label']
    img_path = os.path.join(image_dir, filename)

    if not os.path.exists(img_path):
        print(f"⚠️ Skipping {filename}: file missing.")
        continue

    image = imread(img_path, as_gray=True)

    # Normalize to uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    feats = extract_features(image)
    all_features.append([filename] + feats + [label])

# === Save Features ===
bitplane_cols = [f'{stat}_bitplane{b}' for b in range(8) for stat in ['mean', 'var', 'energy', 'entropy']]
global_cols = ['mean_intensity', 'var_intensity', 'entropy_intensity', 'skewness', 'kurtosis']
residual_cols = ['residual_mean', 'residual_var', 'residual_entropy']
dct_cols = ['dct_mean', 'dct_var', 'dct_entropy']

columns = ['filename'] + bitplane_cols + global_cols + residual_cols + dct_cols + ['label']

df_out = pd.DataFrame(all_features, columns=columns)
df_out.to_csv(output_csv, index=False)
print(f"\n✅ Saved enhanced blind features to: {output_csv}")
