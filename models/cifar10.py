from cleverhans.dataset import CIFAR10

from cleverhans.picklable_model import Conv2D, ReLU, Flatten, Linear, Softmax, Dropout, MLP
from models.cleverhans_utils import MaxPooling2D

from models.model_training import model_training

layers = [
    Conv2D(64, (3, 3), (1, 1), "VALID"),
    ReLU(),
    Conv2D(64, (3, 3), (1, 1), "VALID"),
    ReLU(),
    MaxPooling2D((2, 2), (2, 2), "VALID"),
    Conv2D(128, (3, 3), (1, 1), "VALID"),
    ReLU(),
    Conv2D(128, (3, 3), (1, 1), "VALID"),
    ReLU(),
    MaxPooling2D((2, 2), (2, 2), "VALID"),
    Flatten(),
    Linear(256),
    ReLU(),
    Dropout(),
    Linear(256),
    ReLU(),
    Dropout(),
    Linear(10),
    Softmax()
]

model = MLP(layers, (None, 32, 32, 3))

cifar10 = CIFAR10(train_start=0, train_end=50000, test_start=0, test_end=10000)
x_train, y_train = cifar10.get_set('train')
x_test, y_test = cifar10.get_set('test')

y_train = y_train.reshape((50000, 10))
y_test = y_test.reshape((10000, 10))

model_training(model, "cifar10.joblib", x_train, y_train, x_test, y_test, nb_epochs=20)
