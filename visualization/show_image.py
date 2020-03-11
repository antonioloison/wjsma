"""
Visualise image in different formats
"""
import pandas as pd

import matplotlib.pyplot as plt

first_index = 20
last_index = 30

INDEXES = {"airplane": 30, "automobile": 4, "bird": 57, "cat": 91, "deer": 28, "dog": 27, "frog": 19, "horse": 7,
           "ship": 8, "truck": 14}


def image_square():
    """
    Visualise a square of images and attacks for the cifar10 dataset
    """
    f, axarr = plt.subplots(10, 10)
    f.set_size_inches(16, 6)
    for i, index in enumerate(INDEXES.values()):
        file = r"../attack/cifar10/simple_train/simple_image_" + str(index) + ".csv"
        csv = pd.read_csv(file)
        for j in range(10):
            if i == j:
                img = csv["original_image_" + str(index)][:3072].values.reshape(32, 32, 3)
            else:
                img = csv["number_" + str(index) + "_" + str(i) + "_to_" + str(j)][:3072].values.reshape(32, 32, 3)
            axarr[i, j].imshow(img)
            axarr[i, j].axis('off')
    plt.show()


# image_square()

def one_line(file_path):
    """
    Visualise image line with image for each attack and original image
    :param file_path:
    """
    f, axarr = plt.subplots(1, 10)
    f.set_size_inches(16, 6)
    csv = pd.read_csv(file_path)
    origin_class = int(csv.columns[0][-6])
    image_number = int(file_path[-5])
    if 'mnist' in file_path:
        image_size = 784
        image_shape = (28, 28)
        image_color = 'gray'
    elif 'cifar10' in file_path:
        image_size = 3072
        image_shape = (32, 32, 3)
        image_color = None
    for i in range(10):
        if i == origin_class:
            img = csv["original_image_" + str(image_number)][:image_size].values.reshape(image_shape)
        else:
            img = csv["number_" + str(image_number) + "_" + str(origin_class) + "_to_" + str(i)][
                  :image_size].values.reshape(image_shape)
        axarr[i].imshow(img, image_color)
        axarr[i].axis('off')
    plt.show()


# Single image
def single_image(folder_path, index, target_class):
    """
    Visualise original image, and corresponding jsma and wjsma attacks
    :param folder_path: folder of the attacks
    :param index: image index in folder
    :param target_class: target class of the attack
    """
    f, axarr = plt.subplots(1, 3)
    f.set_size_inches(16, 6)
    if 'mnist' in folder_path:
        image_size = 784
        image_shape = (28, 28)
        image_color = 'gray'
    elif 'cifar10' in folder_path:
        image_size = 3072
        image_shape = (32, 32, 3)
        image_color = None
    file_simple = folder_path + "\\" + "simple_train/simple_image_" + str(index) + ".csv"
    file_weighted = folder_path + "\\" + "weighted_train/weighted_image_" + str(index) + ".csv"
    csv = pd.read_csv(file_simple)
    origin_class = int(csv.columns[0][-6])
    if origin_class == target_class:
        raise ValueError("Same target class as the class predicted by the neural network!")
    for i in range(3):
        if i == 0:
            img = csv["original_image_" + str(index)][:image_size].values.reshape(image_shape)
        elif i == 1:
            img = csv["number_" + str(index) + "_" + str(origin_class) + "_to_" + str(target_class)][
                  :image_size].values.reshape(image_shape)
        elif i == 2:
            csv = pd.read_csv(file_weighted)
            img = csv["number_" + str(index) + "_" + str(origin_class) + "_to_" + str(target_class)][
                  :image_size].values.reshape(image_shape)
        axarr[i].imshow(img, image_color)
        axarr[i].axis('off')
    plt.show()

def main(image_type):
    if image_type == "single":
        simgle_image(r"../attack/cifar10", 1, 8)
    elif image_type == "line":
        one_line(file_path)
    elif image_type == "square":
        image_square()
    else:
        raise ValueError("Wrong image type")
