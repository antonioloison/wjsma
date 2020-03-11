"""
Analysis of the results and comparison between weighted and simple JSMA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('precision', 3)

FOLDER_PATH = r"D:\nn_robustness\mnist_final\targeted\train"
IMAGE_NUMBER = 1
TARGET_CLASS = 6
ORIGIN_CLASS = 0


def prob_length(probabilities):
    """
    Computes index of the length of the probability vector without completing zeros
    :param probabilities: array with listof probabilities with zeros
    :return: length of probabilities until the end of the attack,
    """
    zero_matrix = np.zeros((10,))
    max_length = probabilities.shape[0]
    for i in range(0, max_length, 10):
        if np.array_equal(probabilities[i:(i + 10), 0], zero_matrix):
            return i
    return max_length


def probabilities_array(file_path, target_class):
    """
    Gets the interesting probabilities that will be displayed in graph
    :param file_path: attack file path
    :param target_class: target of the adversarial sample
    :return: array to plot
    """
    if 'mnist' in file_path:
        image_size = 784
    elif 'cifar10' in file_path:
        image_size = 3072
    else:
        raise ValueError("file_path doesn't contain the dataset name, either 'mnist or 'cifar10")
    csv = pd.read_csv(file_path)
    max_length = csv.shape[0] - 3 - image_size
    first_class = np.argmax(csv.iloc[image_size:(image_size + 10), 0].values)
    if target_class < first_class:
        index = target_class
    elif target_class > first_class:
        index = target_class - 1
    else:
        raise ValueError("The target has the same value as the predicted class.")
    probabilities = csv.iloc[image_size:(image_size + max_length), index].values.reshape((max_length, 1))
    prob_len = prob_length(probabilities)
    return probabilities[:prob_len, 0]


def visualise(folder_path=FOLDER_PATH, origin_class=ORIGIN_CLASS, target_class=TARGET_CLASS):
    """
    Shows the graph of the target and origin class probabilities in the jsma and wjsma attacks
    :param folder_path:
    :param origin_class:
    :param target_class:
    :return:
    """
    if "mnist" in folder_path and "ciar10" not in folder_path:
        set_type = "mnist"
    elif "cifar10" in folder_path and "mnist" not in folder_path:
        set_type = "cifar10"
    else:
        raise ValueError("You have to mention the dataset name in the folder path either mnist of cifar10")

    simple_path = folder_path + "\\" + set_type + r"_simple_train\simple_image_" + str(IMAGE_NUMBER) + ".csv"
    weighted_path = folder_path + "\\" + set_type + r"_weighted_train\weighted_image_" + str(IMAGE_NUMBER) + ".csv"
    simple_probs = probabilities_array(simple_path, 6)
    weighted_probs = probabilities_array(weighted_path, 6)

    simple_target_probs = simple_probs[target_class::10]
    weighted_target_probs = weighted_probs[target_class::10]
    simple_origin_probs = simple_probs[origin_class::10]
    weighted_origin_probs = weighted_probs[origin_class::10]

    plt.plot(simple_target_probs, label="JSMA target")
    plt.plot(simple_origin_probs, label="JSMA others")
    plt.plot(weighted_target_probs, label="WJSMA target")
    plt.plot(weighted_origin_probs, label="WJSMA others")

    plt.xlabel('Iterations')
    plt.ylabel('Probabilities')

    plt.legend()
    plt.show()
