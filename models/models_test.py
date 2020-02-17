from cleverhans.dataset import CIFAR10
from cleverhans.utils_tf import model_eval
from cleverhans.serial import load

import tensorflow as tf

sess = tf.Session()

with sess.as_default():
    model = load('joblibs/white_box_cifar10.joblib')

cifar10 = CIFAR10(train_start=0, train_end=50000, test_start=0, test_end=10000)
x_train, y_train = cifar10.get_set('train')
x_test, y_test = cifar10.get_set('test')

img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

eval_params = {'batch_size': 128}

predictions = model.get_logits(x)

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

evaluate()
