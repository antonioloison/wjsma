import sys

sys.path.append('/usr/users/gpupro/gpupro_gzeller/venv/wjsma/models')

from cleverhans.dataset import CIFAR10
from cleverhans.picklable_model import Conv2D, ReLU, Softmax, MLP, GlobalAveragePool
from models.cleverhans_utils import MaxPooling2D

from models.model_training import model_training

layers = [Conv2D(64, (3, 3), (1, 1), "SAME"),
          ReLU(),
          Conv2D(128, (3, 3), (1, 1), "SAME"),
          ReLU(),
          MaxPooling2D((2, 2), (2, 2), "VALID"),
          Conv2D(128, (3, 3), (1, 1), "SAME"),
          ReLU(),
          Conv2D(256, (3, 3), (1, 1), "SAME"),
          ReLU(),
          MaxPooling2D((2, 2), (2, 2), "VALID"),
          Conv2D(256, (3, 3), (1, 1), "SAME"),
          ReLU(),
          Conv2D(512, (3, 3), (1, 1), "SAME"),
          ReLU(),
          MaxPooling2D((2, 2), (2, 2), "VALID"),
          Conv2D(10, (3, 3), (1, 1), "SAME"),
          GlobalAveragePool(),
          Softmax()]

model = MLP(layers, (None, 32, 32, 3))

cifar10 = CIFAR10(train_start=0, train_end=50000, test_start=0, test_end=10000)
x_train, y_train = cifar10.get_set('train')
x_test, y_test = cifar10.get_set('test')

y_train = y_train.reshape((50000, 10))
y_test = y_test.reshape((10000, 10))

model_training(model, "white_box_cifar10.joblib", x_train, y_train, x_test, y_test, nb_epochs=10, batch_size=128,
               learning_rate=.001, label_smoothing=0.1)
