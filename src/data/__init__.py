import os
from os.path import dirname as dirname
from .loggers import *

DATA_DIR = dirname(dirname(dirname(os.path.abspath(__file__))))+'/compressed_dataset'
IMG_HEIGHT = 128 
IMG_WIDTH = 256
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
NUM_CLASSES = 124
