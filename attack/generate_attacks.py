"""
Generates the adversarial samples against the specified model and saves each attack in a single CSV file
"""

import tensorflow as tf

import pandas as pd

import numpy as np

from attack.saliency_map_attack import SaliencyMapMethod

from cleverhans.utils import other_classes
from cleverhans.utils_tf import model_argmax
from cleverhans.serial import load

import os


def generate_attacks(save_path, file_path, x_set, y_set, attack, first_index, last_index):
    """
    Run evaluation on a saved model
    :param save_path: path where attacks will be saved
    :param file_path: path to model to evaluate
    :param x_set: the input tensors
    :param y_set: the output tensors
    :param weighted: boolean representing which version of JSMA you want to test
    :param first_index: the first sample index
    :param last_index: the last sample indexd
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sess = tf.Session()

    img_rows, img_cols, channels = x_set.shape[1:4]
    nb_classes = y_set.shape[1]

    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))

    with sess.as_default():
        model = load(file_path)

    assert len(model.get_params()) > 0

    jsma = SaliencyMapMethod(model, sess=sess)
    jsma_params = {'theta': 1, 'gamma': 0.3,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None, 'attack': attack}

    preds = model(x)

    for sample_ind in range(first_index, last_index):
        results = pd.DataFrame()

        print('Attacking input %i/%i' % (sample_ind + 1, last_index))

        sample = x_set[sample_ind:(sample_ind + 1)]
        current_class = int(np.argmax(y_set[sample_ind]))
        target_classes = other_classes(nb_classes, current_class)

        for target in target_classes:
            one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
            one_hot_target[0, target] = 1
            jsma_params['y_target'] = one_hot_target
            adv_x, predictions = jsma.generate_np(sample, **jsma_params)

            res = int(model_argmax(sess, x, preds, adv_x) == target)

            adv_x_reshape = adv_x.reshape(-1)
            test_in_reshape = x_set[sample_ind].reshape(-1)
            nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
            percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

            results['number_' + str(sample_ind) + '_' + str(current_class) + '_to_' + str(target)] = \
                np.concatenate((adv_x_reshape.reshape(-1),
                                predictions.reshape(-1),
                                np.array([nb_changed, percent_perturb, res]))
                               )

        sample_vector = sample.reshape(-1)
        shape1 = sample_vector.shape[0]
        shape2 = results.shape[0]

        results['original_image_' + str(sample_ind)] = \
            np.concatenate((sample.reshape(-1), np.zeros((shape2 - shape1,))))

        results.to_csv(save_path + '/' + attack + '_image_' + str(sample_ind) + '.csv', index=False)
