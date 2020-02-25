from cleverhans.train import train
from cleverhans.utils_tf import model_eval
from cleverhans.loss import CrossEntropy
from cleverhans.serial import save
from cleverhans.picklable_model import PicklableModel

import tensorflow as tf
import numpy as np


NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001


def model_training(model: PicklableModel, file_name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                   y_test: np.ndarray, nb_epochs: int = NB_EPOCHS, batch_size: int = BATCH_SIZE,
                   learning_rate: int = LEARNING_RATE, num_threads: int = None, label_smoothing: float = 0.1):
    """
    Trains the model with the specified parameters
    :param model: the cleverhans picklable model
    :param file_name: the name of the joblib file
    :param x_train: the input train array
    :param y_train: the output train array
    :param x_test: the input test array
    :param y_test: the output test array
    :param nb_epochs: the number of epoch
    :param batch_size: the size of each batch
    :param learning_rate: the optimizer learning rate
    :param num_threads: the number of threads (None to run on the main thread)
    :param label_smoothing: the amount of smoothing used
    """

    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}

    sess = tf.Session(config=tf.ConfigProto(**config_args))

    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate
    }

    eval_params = {'batch_size': batch_size}

    predictions = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=label_smoothing)

    def do_eval(preds, x_set, y_set, accuracy_type):
        """
        Run the evaluation and print the results.
        """
        acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
        print('%s accuracy on %s examples: %0.4f' % (accuracy_type, 'legitimate', acc))

    def evaluate():
        """
        Run evaluation for the naively trained model on clean examples.
        """

        do_eval(predictions, x_train, y_train, 'Train')
        do_eval(predictions, x_test, y_test, 'Test')

    train(sess, loss, x_train, y_train, evaluate=evaluate, args=train_params, var_list=model.get_params())

    with sess.as_default():
        save("joblibs/" + file_name, model)
