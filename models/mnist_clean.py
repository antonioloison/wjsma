"""
LeNet5 architecture from MNIST
"""

from cleverhans.dataset import MNIST
from cleverhans.picklable_model import Conv2D, ReLU, Flatten, Linear, Softmax, MLP

from models.cleverhans_utils import MaxPooling2D
from models.model_utls import model_training, model_testing


def model_train():
    """
    Creates the joblib of LeNet-5 over the MNIST dataset
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

    mnist = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')

    model_training(model, "mnist_clean.joblib", x_train, y_train, x_test, y_test, nb_epochs=10, batch_size=128,
                   learning_rate=0.001)


def model_test():
    """
    Runs the evaluation and prints out the results
    """

    mnist = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000)
    x_train, y_train = mnist.get_set('train')
    x_test, y_test = mnist.get_set('test')

    model_testing("mnist_clean.joblib", x_train, y_train, x_test, y_test)
