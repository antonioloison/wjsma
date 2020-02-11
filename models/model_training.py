from cleverhans.train import train
from cleverhans.utils_tf import model_eval
from cleverhans.loss import CrossEntropy
from cleverhans.serial import save

import tensorflow as tf


NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001
NB_FILTERS = 64
CLEAN_TRAIN = True


def model_training(model, save_path, x_train, y_train, x_test, y_test, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, num_threads=None, label_smoothing=0.1):
    """
    :param model:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param nb_epochs:
    :param batch_size:
    :param learning_rate:
    :param num_threads:
    :param label_smoothing:
    :return:
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
        print('%s accuracy on %s examples: %0.4f' % (accuracy_type,'legitimate', acc))

    def evaluate():
        """
        Run evaluation for the naively trained model on clean examples.
        """

        do_eval(predictions, x_train, y_train, 'Train')
        do_eval(predictions, x_test, y_test, 'Test')

    train(sess, loss, x_train, y_train, evaluate=evaluate, args=train_params, var_list=model.get_params())

    with sess.as_default():
        save(save_path, model)

        print("Now that the model has been saved, you can evaluate it on the new_jsma_method")
