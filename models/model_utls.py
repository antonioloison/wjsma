from cleverhans.train import train
from cleverhans.utils_tf import model_eval
from cleverhans.loss import CrossEntropy
from cleverhans.serial import save, load

import tensorflow as tf


NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001


def do_eval(session, x, y, predictions, x_set, y_set, params, accuracy_type):
    """
    Runs the evaluation and prints out the results
    :param session: the TF session
    :param x: the input placeholder
    :param y: the output placeholder
    :param predictions: the symbolic logits output of the model
    :param x_set: the input tensors
    :param y_set: the output tensors
    :param params: the evaluation parameters
    :param accuracy_type: either "Train" or "Test"
    :return: 
    """

    accuracy = model_eval(session, x, y, predictions, x_set, y_set, args=params)

    print("%s accuracy: %0.4f" % (accuracy_type, accuracy))


def evaluate(session, x, y, predictions, x_train, y_train, x_test, y_test, params):
    """
    Runs evaluation for the naively trained model on clean examples
    :param session: the TF session
    :param x: the input placeholder
    :param y: the output placeholder
    :param predictions: the symbolic logits output of the model
    :param x_train: the train input tensors
    :param y_train: the train output tensors
    :param x_test: the test input tensors
    :param y_test: the test output tensors
    :param params: the evaluation parameters
    """

    do_eval(session, x, y, predictions, x_train, y_train, params, "Train")
    do_eval(session, x, y, predictions, x_test, y_test, params, "Test")


def model_training(model, file_name, x_train, y_train, x_test, y_test, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE, num_threads=None, label_smoothing=0.1):
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
    
    session = tf.Session(config=tf.ConfigProto(**config_args))

    img_rows, img_cols, channels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    train_params = {
        "nb_epochs": nb_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }

    eval_params = {"batch_size": batch_size}

    predictions = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=label_smoothing)
    
    def train_evaluation():
        """
        Prints the performances of the models after each epoch
        """
        
        evaluate(session, x, y, predictions, x_train, y_train, x_test, y_test, eval_params)

    train(session, loss, x_train, y_train, evaluate=train_evaluation, args=train_params, var_list=model.get_params())

    with session.as_default():
        save("models/joblibs/" + file_name, model)


def model_testing(file_name, x_train, y_train, x_test, y_test):
    """
    Runs the evaluation and prints out the results
    :param file_name: the name of the joblib file
    :param x_train: the input train array
    :param y_train: the output train array
    :param x_test: the input test array
    :param y_test: the output test array
    """

    session = tf.Session()

    with session.as_default():
        model = load("models/joblibs/" + file_name)

    img_rows, img_cols, channels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    eval_params = {"batch_size": 128}

    predictions = model.get_logits(x)

    evaluate(session, x, y, predictions, x_train, y_train, x_test, y_test, eval_params)


