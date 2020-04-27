"""
Generate the adversarial samples against the CIFAR10 model
"""

from cleverhans.dataset import CIFAR10

from attack.generate_attacks import generate_attacks

FILE_NAME = "models/joblibs/cifar10.joblib"


def cifar10_save_attacks(weighted, set_type, first_index, last_index):
    """
    Generate attacks over the AllConvolutional CIFAR10 model
    :param weighted: switches between JSMA and WJSMA
    :param set_type: either "train" or "test"
    :param first_index: the first sample index
    :param last_index: the last sample index
    """

    if weighted:
        attack_type = "wjsma"
    else:
        attack_type = "jsma"

    cifar10 = CIFAR10(train_start=0, train_end=50000, test_start=0, test_end=10000)
    x_set, y_set = cifar10.get_set(set_type)

    y_set = y_set.reshape((y_set.shape[0], 10))

    generate_attacks("attack/cifar10/" + attack_type + "_" + set_type, "models/joblibs/cifar10.joblib",
                     x_set, y_set, weighted, 1, first_index, last_index)
