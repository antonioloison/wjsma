"""
Computation of the statistics of the generated attacks in folder
"""

import pandas
import numpy as np
import os


def average_stat(folder, with_max_threshold=True):
    """
    Prints out the stats of the attack
    :param folder: the csv folder
    :param with_max_threshold: uses the max threshold as the upper limit to compute stats for unsuccessful samples.
    """

    if "mnist" in folder:
        image_size = 784
        max_distortion = 0.145
        max_pixel_number = int(image_size * max_distortion / 2) * 2
    elif "cifar10" in folder:
        image_size = 3072
        max_distortion = 0.037
        max_pixel_number = int(image_size * max_distortion / 2) * 2
    else:
        raise ValueError(
            "Invalid folder name, it must have the name of the dataset somewhere either 'mnist' or 'cifar10'")

    average_distortion = 0
    average_distortion_successful = 0
    average_pixel_number = 0
    average_pixel_number_successful = 0

    total_samples = 0
    total_samples_successful = 0

    for file in os.listdir(folder):
        df = pandas.read_csv(folder + file)
        df_values = df.to_numpy()

        first_class = np.argmax(df_values[image_size:(image_size + 10), 0])
        good_prediction = (int(df.columns[0][-6]) == first_class)

        if not good_prediction:
            continue

        for i in range(9):
            total_samples += 1

            if with_max_threshold:
                average_pixel_number += min(df_values[-3, i], max_pixel_number)
                average_distortion += min(df_values[-2, i], max_distortion)
            else:
                average_pixel_number += df_values[-3, i]
                average_distortion += df_values[-2, i]

            if df_values[-2, i] < max_distortion:
                total_samples_successful += 1

                average_pixel_number_successful += df_values[-3, i]
                average_distortion_successful += df_values[-2, i]

    print(folder)
    print("----------------------")
    print("WELL PREDICTED ORIGINAL SAMPLES:", total_samples)
    print("SUCCESS RATE (MISS CLASSIFIED):", total_samples_successful / total_samples)
    print("AVERAGE NUMBER OF CHANGED PIXELS:", average_pixel_number / total_samples)
    print("AVERAGE DISTORTION:", average_distortion / total_samples)
    print("----------------------")
    print("AVERAGE SUCCESSFUL NUMBER OF CHANGED PIXELS:", average_pixel_number_successful / total_samples_successful)
    print("AVERAGE SUCCESSFUL DISTORTION:", average_distortion_successful / total_samples_successful)
    print("----------------------\n")
