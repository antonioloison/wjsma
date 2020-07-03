"""
Generate images for the MNIST dataset
"""

from cleverhans.dataset import MNIST

from attack.generate_attacks import generate_attacks


def mnist_defense_save_attacks(attack, set_type, first_index, last_index):
    """
    Generate attacks over the LeNet-5 model defended against WJSMA
    :param weighted: switches attack between JSMA and WJSMA
    :param set_type: either "train" or "test"
    :param first_index: the first sample index
    :param last_index: the last sample index
    :return:
    """

    mnist = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000)
    x_set, y_set = mnist.get_set(set_type)

    generate_attacks("defense/mnist_defense_wjsma/" + attack + "_" + set_type,
                     "models/joblibs/mnist_defense_wjsma.joblib",
                     x_set, y_set, attack, first_index, last_index)
