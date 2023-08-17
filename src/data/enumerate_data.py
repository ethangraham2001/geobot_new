import os
from os.path import dirname as dirname


data_dir = dirname(dirname(dirname(os.path.abspath(__file__))))+'/compressed_dataset'
def enumerate_elements_in_subfolders(dir):
    folder_counts = {}

    total = 0
    for root, dirs, files in os.walk(dir):
        subfolder_name = os.path.basename(root)
        subfolder_count = len(files)
        folder_counts[subfolder_name] = subfolder_count
        total += subfolder_count

    return folder_counts, total

subfolder_counts, total = enumerate_elements_in_subfolders(data_dir)

subfolder_counts = sorted(subfolder_counts.items(), key=lambda item: item[0])

print(f"total: {total}")
for subfolder, count in subfolder_counts:
    # count = subfolder_counts[subfolder] 
    print(f"{subfolder}: {(float(count)/total * 100.0):.2f}%")
