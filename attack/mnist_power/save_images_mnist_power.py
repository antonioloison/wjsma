"""
Generate the adversarial samples against the MNIST model
"""

from cleverhans.dataset import MNIST

from attack.generate_attacks import generate_attacks


def mnist_power_save_attacks(weighted, power, set_type, first_index, last_index):
    """
    Generate attacks over LeNet-5 model
    :param weighted: switches between JSMA and WJSMA
    :param power: the power applied to the probabilities in WJSMA mode
    :param set_type: either "train" or "test"
    :param first_index: the first sample index
    :param last_index: the last sample index
    """

    if weighted:
        attack_type = "wjsma"
    else:
        attack_type = "jsma"

    mnist = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000)
    x_set, y_set = mnist.get_set(set_type)

    generate_attacks("attack/mnist_power/" + attack_type + "_" + set_type + "_" + str(power),
                     "models/joblibs/lenet-5.joblib", x_set, y_set, weighted, power, first_index, last_index)
