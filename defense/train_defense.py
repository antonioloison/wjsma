"""
LeNet5 architecture from MNIST trained with the augmented dataset
"""

from cleverhans.dataset import MNIST
from cleverhans.picklable_model import Conv2D, ReLU, Flatten, Linear, Softmax, MLP

from models.cleverhans_utils import MaxPooling2D
from models.model_utls import model_training, model_testing

import numpy as np

TRAIN_START = 0
TRAIN_END = 60000
TEST_START = 0
TEST_END = 10000
AUGMENT_SIZE = 20000
NB_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def model_train(attack_type):
    """
    Creates the joblib of LeNet-5 over the MNIST dataset
    :param attack_type: switches between JSMA, WJSMA and LogAttack defense
    """

    layers = [
        Conv2D(20, (5, 5), (1, 1), "VALID"),
        ReLU(),
        MaxPooling2D((2, 2), (2, 2), "VALID"),
        Conv2D(50, (5, 5), (1, 1), "VALID"),
        ReLU(),
        MaxPooling2D((2, 2), (2, 2), "VALID"),
        Flatten(),
        Linear(500),
        ReLU(),
        Linear(10),
        Softmax()
    ]

    model = MLP(layers, (None, 28, 28, 1))

    mnist = MNIST(train_start=TRAIN_START, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')

    if attack_type == "wjsma":
        x_add = np.load("defense/augmented/x_wjsma.npy")[:AUGMENT_SIZE]
        y_add = np.load("defense/augmented/y_wjsma.npy")[:AUGMENT_SIZE]
    elif attack_type == "jsma":
        x_add = np.load("defense/augmented/x_jsma.npy")[:AUGMENT_SIZE]
        y_add = np.load("defense/augmented/y_jsma.npy")[:AUGMENT_SIZE]
    else:
        x_add = np.load("defense/augmented/x_logattack.npy")[:AUGMENT_SIZE]
        y_add = np.load("defense/augmented/y_logattack.npy")[:AUGMENT_SIZE]

    x_train = np.concatenate((x_train, x_add.reshape(x_add.shape + (1,))), axis=0).astype(np.float32)
    y_train = np.concatenate((y_train, y_add), axis=0).astype(np.float32)

    if attack_type == "wjsma":
        model_training(model, "mnist_defense_wjsma.joblib", x_train, y_train, x_test, y_test, nb_epochs=NB_EPOCHS,
                       batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    elif attack_type == "jsma":
        model_training(model, "mnist_defense_jsma.joblib", x_train, y_train, x_test, y_test, nb_epochs=NB_EPOCHS,
                       batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    else:
        model_training(model, "mnist_defense_logattack.joblib", x_train, y_train, x_test, y_test, nb_epochs=NB_EPOCHS,
                       batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

def model_test(attack_type):
    """
    Runs the evaluation and prints out the results
    :param attack_type: switches between JSMA, WJSMA and LogAttack defense
    """

    mnist = MNIST(train_start=TRAIN_START, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')

    print("ORIGINAL MNIST TEST")

    if attack_type == "wjsma":
        model_testing("mnist_defense_wjsma.joblib", x_train, y_train, x_test, y_test)
    elif attack_type == "jsma":
        model_testing("mnist_defense_jsma.joblib", x_train, y_train, x_test, y_test)
    else:
        model_testing("mnist_defense_logattack.joblib", x_train, y_train, x_test, y_test)

    if attack_type == "wjsma":
        x_add = np.load("defense/augmented/x_wjsma.npy")[:AUGMENT_SIZE]
        y_add = np.load("defense/augmented/y_wjsma.npy")[:AUGMENT_SIZE]
    elif attack_type == "jsma":
        x_add = np.load("defense/augmented/x_jsma.npy")[:AUGMENT_SIZE]
        y_add = np.load("defense/augmented/y_jsma.npy")[:AUGMENT_SIZE]
    else:
        x_add = np.load("defense/augmented/x_logattack.npy")[:AUGMENT_SIZE]
        y_add = np.load("defense/augmented/y_logattack.npy")[:AUGMENT_SIZE]

    x_train = np.concatenate((x_train, x_add.reshape(x_add.shape + (1,))), axis=0).astype(np.float32)
    y_train = np.concatenate((y_train, y_add), axis=0).astype(np.float32)

    print("====================")
    print("AUGMENTED MNIST TEST")

    if attack_type == "wjsma":
        model_testing("mnist_defense_wjsma.joblib", x_train, y_train, x_test, y_test)
    elif attack_type == "jsma":
        model_testing("mnist_defense_jsma.joblib", x_train, y_train, x_test, y_test)
    else:
        model_testing("mnist_defense_logattack.joblib", x_train, y_train, x_test, y_test)
