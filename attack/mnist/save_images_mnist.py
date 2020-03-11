"""
Generate the adversarial samples against the MNIST model
"""

from cleverhans.dataset import MNIST

from attack.generate_attacks import generate_attacks

TRAIN_START = 0
TRAIN_END = 60000
TEST_START = 0
TEST_END = 10000


def mnist_save_attacks(weighted, set_type, first_index, last_index):
    """
    Generate attacks over LeNet-5 model
    :param weighted: switches between JSMA and WJSMA
    :param set_type: either "train" or "test"
    :param first_index: the first sample index
    :param last_index: the last sample index
    :return:
    """

    if weighted:
        attack_type = "weighted"
    else:
        attack_type = "simple"

    mnist = MNIST(train_start=TRAIN_START, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END)
    x_set, y_set = mnist.get_set(set_type)

    generate_attacks("attack/mnist/" + attack_type + "_" + set_type, "models/joblibs/lenet-5.joblib",
                     x_set, y_set, weighted, first_index, last_index)
