import os
import numpy as np
import pandas as pd
from skimage.io import imread
from scipy.stats import entropy, skew, kurtosis
from tqdm import tqdm

attacked_dir = 'D:\CapstoneV2\DataSets\COCOpng2017Sample'      
clean_dir = 'D:\CapstoneV2\DataSets\COCOTrainingImagespng2017'                
metadata_csv = 'D:\CapstoneV2\Metadata\csv\stego_metadata.csv'
output_csv = 'MetaData/csv/diff_features.csv'



def extract_features(image_gray):
    features = []


    for bit in range(8):
        bitplane = ((image_gray >> bit) & 1).astype(np.uint8)
        mean_bp = np.mean(bitplane)
        var_bp = np.var(bitplane)
        energy_bp = np.sum(np.square(bitplane))
        counts = np.bincount(bitplane.flatten(), minlength=2)
        ent_bp = entropy(counts + 1e-8)
        features.extend([mean_bp, var_bp, energy_bp, ent_bp])

    image_flat = image_gray.flatten()
    mean_int = np.mean(image_flat)
    var_int = np.var(image_flat)
    counts = np.bincount(image_flat, minlength=256)
    ent_int = entropy(counts + 1e-8)
    skewness = skew(image_flat)
    kurt = kurtosis(image_flat)
    features.extend([mean_int, var_int, ent_int, skewness, kurt])

    return np.array(features)


metadata = pd.read_csv(metadata_csv)



all_features = []

print("Extracting DIFFERENTIAL features...")

for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
    filename = row['Filename']
    label = row['Label']

    attacked_img_path = os.path.join(attacked_dir, filename)
    clean_img_path = os.path.join(clean_dir, filename)

    if not (os.path.exists(attacked_img_path) and os.path.exists(clean_img_path)):
        print(f"Skipping {filename}: missing clean or attacked version.")
        continue

  
    attacked_img = imread(attacked_img_path, as_gray=True)
    clean_img = imread(clean_img_path, as_gray=True)

   
    if attacked_img.max() <= 1.0:
        attacked_img = (attacked_img * 255).astype(np.uint8)
    else:
        attacked_img = attacked_img.astype(np.uint8)

    if clean_img.max() <= 1.0:
        clean_img = (clean_img * 255).astype(np.uint8)
    else:
        clean_img = clean_img.astype(np.uint8)

 
    attacked_feats = extract_features(attacked_img)
    clean_feats = extract_features(clean_img)
    
    diff_feats = attacked_feats - clean_feats

    all_features.append([filename] + diff_feats.tolist() + [label])



bitplane_cols = [f'{stat}_bitplane{b}' for b in range(8) for stat in ['mean', 'var', 'energy', 'entropy']]
global_cols = ['mean_intensity', 'var_intensity', 'entropy_intensity', 'skewness', 'kurtosis']
diff_cols = [f'diff_{col}' for col in (bitplane_cols + global_cols)]

columns = ['filename'] + diff_cols + ['label']

df_out = pd.DataFrame(all_features, columns=columns)
df_out.to_csv(output_csv, index=False)
print(f"Saved differential features to {output_csv}")
