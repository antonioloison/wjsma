"""
Several test to verify experiments
"""

import pandas as pd

import numpy as np

import os


def different_images(folder_path):
    """
    Verify if all original images of the attacks are different
    :param folder_path: path of the folder containg the dataframes of the attacks
    :return: if all different, True else False and place of the difference
    """
    main_counter = 0
    for root, dir, files1 in os.walk(folder_path):
        for file1 in files1:
            csv1 = pd.read_csv(folder_path + '\\' + file1)
            image1 = csv1[original_column(csv1)].values
            sub_counter = 0
            for root, dir, files2 in os.walk(folder_path):
                for file2 in files2:
                    if sub_counter > main_counter:
                        csv2 = pd.read_csv(folder_path + '\\' + file2)
                        image2 = csv2[original_column(csv2)].values
                        if np.array_equal(image1, image2):
                            return False, (main_counter, sub_counter)
                    sub_counter += 1
            main_counter += 1
    return True

def original_column(csv):
    """
    Returns the name of the column containing the original image that was attacked
    :param csv: Dataframe containing the attack
    :return: name of the column containing the original image
    """
    columns = csv.columns
    for column in columns:
        if 'original' in column:
            return column

def same_originals(folder_path_1, folder_path_2):
    """
    Test if correspondant images in folder are
    :param folder_path_1: path of first folder
    :param folder_path_2: path of second folder
    :return:
    """
    main_counter = 0
    length1 = len(next(os.walk(folder_path_1))[2])
    length2 = len(next(os.walk(folder_path_2))[2])
    for i in range(min(length1,length2)):
        pass
