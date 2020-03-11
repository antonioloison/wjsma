"""
Generate images for the MNIST dataset
"""

from cleverhans.dataset import MNIST

from attack.generate_attacks import generate_attacks


def mnist_defense_save_attacks(weighted, set_type, first_index, last_index):
    """
    Generate attacks over the LeNet-5 model defended against WJSMA
    :param weighted: switches attack between JSMA and WJSMA
    :param set_type: either "train" or "test"
    :param first_index: the first sample index
    :param last_index: the last sample index
    :return:
    """

    if weighted:
        attack_type = "weighted"
    else:
        attack_type = "simple"

    mnist = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000)
    x_set, y_set = mnist.get_set(set_type)

    generate_attacks("defense/mnist_defense_weighted/" + attack_type + "_" + set_type,
                     "models/joblibs/mnist_defense_weighted.joblib",
                     x_set, y_set, weighted, first_index, last_index)
