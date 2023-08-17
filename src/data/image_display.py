import matplotlib.pyplot as plt
import numpy as np

from colorama import Fore, Style
from .loggers import log_success

class ImageDisplay:
    """
    Handles the displaying of images
    Works for torch.tensor as well as np.array
    """

    log_success('Image Display created')

    def display_image(self, img):
        """
        Displays image 
        """

        if type(img) != (type(np.zeros((2)))):
            print(Style.DIM + '--> converting to ndarray' + Style.RESET_ALL)
            img = img.numpy()

        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.show()
