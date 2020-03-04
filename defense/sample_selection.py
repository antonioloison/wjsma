import pandas
import random
import numpy

COUNT = 2000


def generate_extra_set(set_type, weighted):
    """
    Generates the extra dataset
    :param set_type: either train or test
    :param weighted: switches between JSMA and WJSMA samples
    """

    samples = [[] for _ in range(10)]

    if weighted:
        path = "white_box/mnist/weighted_" + set_type + "/weighted_image_"
    else:
        path = "white_box/mnist/simple_" + set_type + "/simple_image_"

    for index in range(10000):
        df = pandas.read_csv(path + str(index) + ".csv")
        np = df.to_numpy()

        label = int(df.columns[0][-6])

        for i in range(9):
            if np[1955, i] < 0.145:
                samples[label].append(np[:784, i].reshape((28, 28)))

        print([len(samples[k]) for k in range(10)])

    x_set = []

    for k in range(10):
        random.shuffle(samples[k])

        samples[k] = samples[k][:COUNT]

        for sample in samples[k]:
            x_set.append([sample, one_hot(k)])

    random.shuffle(x_set)

    x = numpy.array([x_set[k][0] for k in range(len(x_set))])
    y = numpy.array([x_set[k][1] for k in range(len(x_set))])

    if weighted:
        old_x = numpy.load("defense/augmented/x_weighted.npy")
        old_y = numpy.load("defense/augmented/y_weighted.npy")

        x = numpy.concatenate((old_x, x), axis=0)
        y = numpy.concatenate((old_y, y), axis=0)

        numpy.save("defense/augmented/x_weighted.npy", x)
        numpy.save("defense/augmented/y_weighted.npy", y)
    else:
        numpy.save("defense/augmented/x_simple.npy", x)
        numpy.save("defense/augmented/y_simple.npy", y)


def one_hot(index):
    """
    Returns the one hot vector of the specified index
    :param index: the 1 position
    :return: the on hot vector
    """

    vector = numpy.zeros(10)
    vector[index] = 1

    return vector
