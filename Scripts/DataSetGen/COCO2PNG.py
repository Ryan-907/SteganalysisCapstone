import os
from PIL import Image

source_dir = "/CapstoneV2/DataSets/COCOjpg2017"
destination_dir = "/CapstoneV2/DataSets/COCOpng2017"

jpegs = os.listdir(source_dir)  # Get list of files
total = len(jpegs)


for i, im in enumerate(jpegs, 1):
    image = Image.open(os.path.join(source_dir, im))
    im = im.replace('.jpg', '')  # Avoid 'file.jpg.png'
    image.save(os.path.join(destination_dir, f"{im}.png"))

    percent = (i / total) * 100
    print(f"\rProgress: {percent:.2f}% ({i}/{total})", end='')  # Overwrites the same line

print(f"\nConversion complete. {total} images saved to {destination_dir}.")


'''
This converts the COCO dataset 2017 into PNGs. This simplifies stego techniques and makes LSB in particular
more doable. It also ensures consistent file types used.
'''