"""
Computation of the statistics of the generated attacks in folder
"""

import pandas
import os


def average_stat(folder, with_max_threshold=True):
    """
    Prints out the stats of the attack
    :param folder: the csv folder
    """
    if "mnist" in folder:
        image_size = 784
        max_distortion = 0.145
        max_iter = int(image_size * max_distortion / 200)
    elif "cifar10" in folder:
        image_size = 3072
        max_distortion = 0.037
        max_iter = int(image_size * max_distortion / 200)
    else:
        raise ValueError(
            "Invalid folder name, it must have the name of the dataset somewhere either 'mnist' or 'cifar10'")

    average_distortion = 0
    average_distortion_successful = 0
    average_iteration = 0
    average_iteration_successful = 0

    total_samples = 0
    total_samples_successful = 0

    for file in os.listdir(folder):
        df = pandas.read_csv(folder + file)
        np = df.to_numpy()

        first_class = np.argmax(df.iloc[image_size:(image_size + 10), 0].values)
        good_prediction = (int(df.columns[0][-6]) == first_class)

        if not good_prediction:
            continue

        for i in range(9):
            total_samples += 1

            if with_max_threshold:
                average_iteration += min(np[-3, i], max_iter)
                average_distortion += min(np[-2, i], max_distortion)
            else:
                average_iteration += np[-3, i]
                average_distortion += np[-2, i]

            if np[-2, i] < max_distortion:
                total_samples_successful += 1

                average_iteration_successful += np[-3, i]
                average_distortion_successful += np[-2, i]

    print(folder)
    print("----------------------")
    print("WELL PREDICTED ORIGINAL SAMPLES:", total_samples)
    print("SUCCESS RATE (MISS CLASSIFIED):", total_samples_successful / total_samples)
    print("AVERAGE ITERATION:", average_iteration / total_samples)
    print("AVERAGE DISTORTION:", average_distortion / total_samples)
    print("----------------------")
    print("AVERAGE SUCCESSFUL ITERATION:", average_iteration_successful / total_samples_successful)
    print("AVERAGE SUCCESSFUL DISTORTION:", average_distortion_successful / total_samples_successful)
    print("----------------------\n")
