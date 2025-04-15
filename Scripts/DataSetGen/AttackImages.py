import os
import random
import pandas as pd
import cv2
from shutil import copyfile
from itertools import cycle

from Techniques import embed_dct, embed_lsb, DELIMITER

payload_length = cycle(['sentence', 'paragraph', 'full'])


source_dir = "DataSets/COCOpng2017Sample"
output_dir = "DataSets/CCOCOTrainingImagespng2017"
csv_filename = "stego_metadata.csv"
texts_path = "metadata/texts"


train_ratio = 0.8
stego_ratio = 0.66

sample = os.listdir(source_dir)

random.shuffle(sample)
train_size = int(len(sample) * train_ratio)
train_images = sample[:train_size]
test_images = sample[train_size:]


csv_data = []

def select_text(path: str, delim, length: str = None) -> tuple:
    files = [f for f in os.listdir(path) if f.lower().endswith('.txt')]
    if not files:
        raise FileNotFoundError("No text files found in the specified directory.")

    selected_file = os.path.join(path, random.choice(files))
    with open(selected_file, 'r', encoding='utf-8') as file:
        content = file.read()

    if not content:
        raise ValueError(f"File {selected_file} is empty.")

    content = content.replace('\n', ' ').strip()
    if not length:
        length = next(payload_length)

    if length == 'sentence':
        max_start = max(0, len(content) - 20)
        start_index = random.randint(0, max_start) if max_start > 0 else 0
        text = content[start_index:start_index + 20] + delim
    elif length == 'paragraph':
        max_start = max(0, len(content) - 200)
        start_index = random.randint(0, max_start) if max_start > 0 else 0
        text = content[start_index:start_index + 200] + delim
    else:
        text = content + delim

    print(f"\nSelected text preview: {text[:50]}...")
    return text, selected_file, length

def process_images(image_list, split):
    print(f"Processing {split} images... ({len(image_list)} total)")

    for idx, img_name in enumerate(image_list):
        img_path = os.path.join(source_dir, img_name)
        base_name = os.path.splitext(img_name)[0]

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠ Warning: Could not read {img_name}")
            continue

        attack_decision = random.choices(["None", "LSB", "DCT"], [1 - stego_ratio, stego_ratio / 2, stego_ratio / 2])[0]

        if attack_decision == "None":
            dest_path = os.path.join(output_dir, img_name)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            copyfile(img_path, dest_path)
            csv_data.append([img_name, 0, '-', '-'])

        else:
            text_payload, text_source, text_type = select_text(texts_path, DELIMITER)

            if attack_decision == "LSB":
                stego_img, _ = embed_lsb(img, text_payload)
                Label = 1
            elif attack_decision == "DCT":
                stego_img, _ = embed_dct(img, text_payload)
                Label = 2

            dest_path = os.path.join(output_dir, img_name)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            cv2.imwrite(dest_path, stego_img)
            csv_data.append([img_name, Label, text_source, text_type])

        if (idx + 1) % 100 == 0:
            print(f"  → Processed {idx + 1}/{len(image_list)} images...")

process_images(train_images, "Train")
process_images(test_images, "Test")


df = pd.DataFrame(csv_data, columns=["Filename", "Label", "Text Source", "Text Type"])
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, csv_filename), index=False)

print(f"Dataset generation complete! CSV saved at {output_dir}/{csv_filename}")