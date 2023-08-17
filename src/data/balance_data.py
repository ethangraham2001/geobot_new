import os
from os.path import dirname as dirname
import random
data_dir = dirname(dirname(dirname(os.path.abspath(__file__))))+'/compressed_dataset/'+"France"

print(data_dir)


def balace_data(path):
    image_files = [f for f in os.listdir(path)]
    num_images_to_delete = int(len(image_files) * 2/3)

    images_to_delete = random.sample(image_files, num_images_to_delete)


    for image in images_to_delete:
        image_path = os.path.join(path, image)
        os.remove(image_path)
        print(f"Deleted: {image_path}")

balace_data(data_dir) 