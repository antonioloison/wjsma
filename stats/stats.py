import pandas
import os


def average_stat(folder):
    """
    Prints out the stats of the attack
    :param folder: the csv folder
    """

    average_distortion = 0
    average_distortion_successful = 0
    average_iteration = 0
    average_iteration_successful = 0

    total_samples = 0
    total_samples_successful = 0

    for file in os.listdir(folder):
        df = pandas.read_csv(folder + file)
        np = df.to_numpy()

        for i in range(9):
            total_samples += 1

            average_iteration += np[1954, i]
            average_distortion += np[1955, i]

            if np[1956, i] == 1.0:
                total_samples_successful += 1

                average_iteration_successful += np[1954, i]
                average_distortion_successful += np[1955, i]

    print("----------------------")
    print("SUCCESS RATE (MISS CLASSIFIED):", total_samples_successful / total_samples)
    print("AVERAGE ITERATION:", average_iteration / total_samples)
    print("AVERAGE DISTORTION:", average_distortion / total_samples)
    print("----------------------")
    print("AVERAGE SUCCESSFUL ITERATION:", average_iteration_successful / total_samples_successful)
    print("AVERAGE SUCCESSFUL DISTORTION:", average_distortion_successful / total_samples_successful)
    print("----------------------")
