"""
Create extra dataset to train a defended model
"""
import pandas
import random
import numpy
import os

SAMPLE_COUNT = 2000


def generate_extra_set(set_type, attack, sample_per_class=SAMPLE_COUNT):
    """
    Generates the extra dataset
    :param set_type: either train or test
    :param weighted: switches between JSMA and WJSMA samples
    :param sample_per_class: the number of adversarial samples per class
    """

    samples = [[] for _ in range(10)]

    path = "attack/mnist/" + attack + "_" + set_type + "/"

    for file in os.listdir(path):
        df = pandas.read_csv(path + file)
        np = df.to_numpy()

        label = int(df.columns[0][-6])

        for i in range(9):
            if np[785, i] < 0.145:
                samples[label].append(np[:784, i].reshape((28, 28)))

    x_set = []

    for k in range(10):
        random.shuffle(samples[k])

        if len(samples[k]) < sample_per_class:
            raise ValueError("Not enough samples to create the augmented dataset")

        samples[k] = samples[k][:sample_per_class]

        for sample in samples[k]:
            x_set.append([sample, _one_hot(k)])

    random.shuffle(x_set)

    x = numpy.array([x_set[k][0] for k in range(len(x_set))])
    y = numpy.array([x_set[k][1] for k in range(len(x_set))])

    if not os.path.exists("defense/augmented/"):
        os.mkdir("defense/augmented/")

    numpy.save("defense/augmented/" + attack + "_x.npy", x)
    numpy.save("defense/augmented/" + attack + "_y.npy", y)


def _one_hot(index):
    vector = numpy.zeros(10)
    vector[index] = 1

    return vector
