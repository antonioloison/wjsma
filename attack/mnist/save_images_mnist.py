"""
Generate the adversarial samples against the MNIST model
"""

from cleverhans.dataset import MNIST

from attack.generate_attacks import generate_attacks


def mnist_save_attacks(attack_type, set_type, first_index, last_index):
    """
    Generate attacks over LeNet-5 model
    :param attack_type: switches between JSMA and WJSMA and LogAttack
    :param set_type: either "train" or "test"
    :param first_index: the first sample index
    :param last_index: the last sample index
    """

    mnist = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000)
    x_set, y_set = mnist.get_set(set_type)

    generate_attacks("attack/mnist/" + attack_type + "_" + set_type, "models/joblibs/lenet-5.joblib",
                     x_set, y_set, attack_type, first_index, last_index)
