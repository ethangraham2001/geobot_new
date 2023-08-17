import os

from os.path import dirname as dirname

import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from colorama import Fore, Back, Style

from .loggers import log_success, log_failure, log_small
from . import IMG_HEIGHT, IMG_WIDTH, DATA_DIR

class DataLoader:
    """
    Loads images from the '/compressed_dataset' directory,
    which should be placed in the root directory of the project

    """

    data_dir = DATA_DIR
    img_height = IMG_HEIGHT
    img_width = IMG_WIDTH
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(
        root=data_dir,
        transform=transform
    )

    # 60% train, 20% validation, 20% test
    dataset_lengths = [0.6, 0.2, 0.2]

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset=dataset,
        lengths=dataset_lengths
    )

    batch_size = 64
    log_success('DataLoader created')

    def get_loaders(self) -> torch.utils.data.DataLoader:
        """
        returns loaders for all datasets
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)       
        return train_loader, val_loader, test_loader

    def summarize(self):
        print(Style.DIM)
        print("DataLoader Summary:")
        print(f"--> dir = {self.data_dir}")
        print(f"--> batch_size = {self.batch_size}")
        print(f"--> dataset_lengths = {self.dataset_lengths}")
        print(f"--> image dimensions = [{self.img_height}, {self.img_width}]")
        print()
        print(Style.RESET_ALL)
