import os
from math import floor
import shutil

source_dir = "DataSets/COCOpng2017"
destination_dir = "DataSets/TrainingImages"

os.makedirs(destination_dir, exist_ok= True)

full_set = os.listdir(source_dir)
subset = full_set[::len(full_set) // floor(len(full_set) *.1)] #Selects 10 percent of a list. Example 100// 100*.1 = 10 items, or 10%. 

for file_name in subset:
    src_path = os.path.join(source_dir, file_name)
    dst_path = os.path.join(destination_dir, file_name)
    shutil.move(src_path, dst_path)
print(len(os.listdir(destination_dir)))