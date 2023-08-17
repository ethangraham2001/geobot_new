import os
from . import DATA_DIR

def nbr_to_country(label: int) -> str:
    """
    Turns the country's index in file structure into name 
    :param num: label of the country
    """

    directories = sorted(os.listdir(DATA_DIR))
    return directories[label]