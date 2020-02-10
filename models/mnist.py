"""
LeNet5 architecture
"""

from cleverhans.dataset import MNIST
from cleverhans.picklable_model import Conv2D, ReLU, Flatten, Linear, Softmax, MLP
from utils.cleverhans_utils import MaxPooling2D

from models.model_training import model_training


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

model_training(model, x_train, y_train, x_test, y_test)
