"""
Generate the adversarial samples against the CIFAR10 model
"""

from cleverhans.dataset import CIFAR10

from attack.generate_attacks import generate_attacks

FILE_NAME = "models/joblibs/cifar10.joblib"
TRAIN_START = 0
TRAIN_END = 50000
TEST_START = 0
TEST_END = 10000


def cifar10_save_attacks(weighted, set_type, first_index, last_index):
    """
    Generate attacks over the AllConvolutional CIFAR10 model
    :param weighted: switches between JSMA and WJSMA
    :param set_type: either "train" or "test"
    :param first_index: the first sample index
    :param last_index: the last sample index
    """

    if weighted:
        attack_type = "weighted"
    else:
        attack_type = "simple"

    cifar10 = CIFAR10(train_start=TRAIN_START, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END)
    x_set, y_set = cifar10.get_set(set_type)

    y_set = y_set.reshape((y_set.shape[0], 10))

    generate_attacks("attack/cifar10/" + attack_type + "_" + set_type, "models/joblibs/cifar10.joblib",
                     x_set, y_set, weighted, first_index, last_index)
