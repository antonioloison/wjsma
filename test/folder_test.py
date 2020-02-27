"""
Several test to verify experiments
"""

import pandas as pd

import numpy as np

import os

from cleverhans.dataset import MNIST, CIFAR10


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

def different_images(folder_path):
    """
    Verify if all original images of the attacks are different
    :param folder_path: path of the folder containg the dataframes of the attacks
    :return: if all different, True else False and place of the difference
    """
    counter = 0
    image_list = []
    for root, dir, files1 in os.walk(folder_path):
        for file1 in files1:
            if counter % 1000 == 0:
                print(counter)
            csv1 = pd.read_csv(folder_path + '\\' + file1)
            image1 = csv1[original_column(csv1)].values
            reshaped_image = image1.reshape(-1)
            tuple_image = tuple(reshaped_image)
            image_list.append(tuple_image)
            counter += 1
    return len(image_list) == len(set(image_list))

# print(different_images(r"D:\nn_robustness\le_net_5_final\targeted\test\simple_test"))

def same_originals(folder_path_1, folder_path_2):
    """
    Test if corespondent images in folder are
    :param folder_path_1: path of first folder
    :param folder_path_2: path of second folder
    :return: True if same originals, False and index where the difference is if immages are different
    """
    length1 = len(next(os.walk(folder_path_1))[2])
    length2 = len(next(os.walk(folder_path_2))[2])
    def attack_type(path):
        if 'simple' in path:
            return 'simple'
        elif 'weighted' in path:
            return 'weighted'
    attack_type1, attack_type2 = attack_type(folder_path_1), attack_type(folder_path_2)
    for i in range(min(length1,length2)):
        if i % 1000 == 0:
            print(i)
        csv1 = pd.read_csv(folder_path_1 + '\\' + attack_type1 + '_image_' + str(i) + '.csv')
        csv2 = pd.read_csv(folder_path_2 + '\\' + attack_type2 + '_image_' + str(i) + '.csv')
        image1 = csv1[original_column(csv1)]
        image2 = csv2[original_column(csv2)]
        if not(np.array_equal(image1,image2)):
            return False, i
    return True

# print(same_originals(r"D:\nn_robustness\le_net_5_final\targeted\test\simple_test",
#                      r"D:\nn_robustness\le_net_5_final\targeted\test\weighted_test"))

def all_the_same(folder_path_1, folder_path_2):
    """
    Test if corespondent images in folder are
    :param folder_path_1: path of first folder
    :param folder_path_2: path of second folder
    :return: True if same originals, False and index where the difference is if immages are different
    """
    length1 = len(next(os.walk(folder_path_1))[2])
    length2 = len(next(os.walk(folder_path_2))[2])
    def attack_type(path):
        if 'simple' in path:
            return 'simple'
        elif 'weighted' in path:
            return 'weighted'
    attack_type1, attack_type2 = attack_type(folder_path_1), attack_type(folder_path_2)
    for i in range(min(length1,length2)):
        csv1 = pd.read_csv(folder_path_1 + '\\' + attack_type1 + '_image_' + str(i) + '.csv')
        csv2 = pd.read_csv(folder_path_2 + '\\' + attack_type2 + '_image_' + str(i) + '.csv')
        image1 = csv1.values
        image2 = csv2.values
        if not(np.array_equal(image1,image2)):
            return False, i
    return True

# print(same_originals(r"C:\Users\Antonio\Projects\wjsma\white_box\mnist\simple_test",
#                      r"C:\Users\Antonio\Projects\wjsma\white_box\mnist\mnist_2_simple_test"))

def same_originals_to_dataset(folder_path, set_type, dataset):
    """
    Test if corespondent images in folder are
    :param folder_path_1: path of first folder
    :param folder_path_2: path of second folder
    :return: True if same originals, False and index where the difference is if immages are different
    """
    length = len(next(os.walk(folder_path))[2])
    if dataset == 'mnist':
        x_set, y_set = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000).get_set(set_type)
        image_size = 784
    if dataset == 'cifar10':
        x_set, y_set = CIFAR10(train_start=0, train_end=60000, test_start=0, test_end=10000).get_set(set_type)
        image_size = 3072
    def attack_type(path):
        if 'simple' in path:
            return 'simple'
        elif 'weighted' in path:
            return 'weighted'
    attack_type = attack_type(folder_path)
    for i in range(length):
        if i % 1000 == 0:
            print(i)
        csv = pd.read_csv(folder_path + '\\' + attack_type + '_image_' + str(i) + '.csv')
        image1 = csv[original_column(csv)].values[:image_size]
        image2 = x_set[i:(i+1)].reshape(-1)
        if not(np.allclose(image1,image2)):
            return False, i
    return True

# print(same_originals_to_dataset(r"D:\nn_robustness\le_net_5_final\targeted\train\mnist_simple_train", 'train', 'mnist'))
