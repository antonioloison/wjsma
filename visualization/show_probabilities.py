"""
Analysis of the results and comparison between weighted and simple JSMA
"""

import os
import pandas as pd

pd.set_option('precision', 3)
import numpy as np
import time
import matplotlib.pyplot as plt

IMAGE_SIZE = 784

simple_path_train = r"D:\nn_robustness\le_net_5_final\targeted\train\mnist_simple_train"
MAX_MAX_ITER = 117
MAXIMUM_DISTORTION = 14.5
weighted_path_train = r"D:\nn_robustness\le_net_5_final\targeted\train\mnist_weighted_train"

def probabilities_visualization(origin_class, target_class, folder_path, maximum_distortion=30, well_predicted=True):
    counter = 0
    new_counter = 0
    bad_preds = 0
    probs_size = 80 * 10
    total_probs = np.zeros((0, 0))
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if new_counter > 0:
                break
            if counter % 1000 == 0:
                print(counter)
            df = pd.read_csv(folder_path + '\\' + file)
            first_class = np.argmax(df.iloc[IMAGE_SIZE:(IMAGE_SIZE + 10), 0].values)
            origin = int(df.columns[0][-6])
            if well_predicted:
                good_prediction = (int(df.columns[0][-6]) == first_class)
            else:
                good_prediction = True
            if good_prediction and origin == origin_class:
                index = -1
                for i in range(10):
                    if i < first_class:
                        index = i
                    elif i > first_class:
                        index = i - 1
                    if index != -1:
                        target = int(df.columns[index][-1])
                        if target_class == target:
                            if new_counter == 0:
                                total_probs = df.iloc[IMAGE_SIZE:(IMAGE_SIZE + probs_size), index].values.reshape(
                                    ((probs_size, 1)))
                            else:
                                probabilities = df.iloc[IMAGE_SIZE:(IMAGE_SIZE + probs_size), index].values.reshape(
                                    ((probs_size, 1)))
                                total_probs = np.concatenate((total_probs, probabilities), axis=1)
                            print(file)
                            new_counter += 1
                    index = -1
            else:
                bad_preds += 1
            counter += 1
        print(total_probs.shape)
        mean_probs = np.mean(total_probs, axis=1)
        print(mean_probs.shape)
    return counter, bad_preds, mean_probs

TARGET = 6
simple_probs = probabilities_visualization(0, TARGET, simple_path_train)[2]
weighted_probs = probabilities_visualization(0, TARGET, weighted_path_train)[2]

def get_arrays(mean_values, target_class, maximum_distortion=30):
    probs_size = 80
    target_probs = []
    other_probs = []
    for i in range(probs_size):
        sum = 0
        for j in range(10):
            index = 10 * i + j
            if j == target_class:
                target_probs.append(mean_values[index])
            else:
                sum += mean_values[index]
        other_probs.append(sum)
    return target_probs, other_probs


simple_t_probs, simple_o_probs = get_arrays(simple_probs, 0, maximum_distortion=30)
weighted_t_probs, weighted_o_probs = get_arrays(weighted_probs, 0, maximum_distortion=30)

def vizualize(simple_t_probs, simple_o_probs, weighted_t_probs, weighted_o_probs):
    plt.plot(simple_t_probs[:simple_t_probs.index(0)], label="simple target")
    print([(i,simple_t_probs[:simple_t_probs.index(0)][i]) for i in range(simple_t_probs.index(0))])
    # plt.plot(simple_o_probs[:simple_o_probs.index(0)], label="simple others")
    plt.plot(weighted_t_probs[:weighted_t_probs.index(0)], label="weighted target")
    print([(i,weighted_t_probs[:weighted_t_probs.index(0)][i]) for i in range(weighted_t_probs.index(0))])

    # plt.plot(weighted_o_probs[:weighted_o_probs.index(0)], label="weighted others")

    plt.xlabel('Iterations')
    plt.ylabel('Probabilities')

    plt.legend()
    plt.show()

vizualize(simple_t_probs, simple_o_probs, weighted_t_probs, weighted_o_probs)
