"""
This script evaluates trained models that have been saved to the filesystem and saves each attack in one csv file
per image attacked
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

import pandas as pd

import numpy as np

from saliency_map_attack import SaliencyMapMethod

from cleverhans.utils import other_classes
from cleverhans.utils_tf import silence, model_argmax
from cleverhans.serial import load
silence()

def generate_attacks(save_path, filepath, set_type, dataset, weighted,
                   train_start=0, train_end=60000, test_start=0,
                   test_end=10000):
    """
    Run evaluation on a saved model
    :param save_path: path where attacks will be saved
    :param filepath: path to model to evaluate
    :param set_type: 'test' or 'train' depending on which dataset you want to test the attack
    :param weighted: boolean representing which version of JSMA you want to test
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param batch_size: size of evaluation batches
    """

    sess = tf.Session()

    # Get MNIST test data
    dataset = dataset(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)

    x_set, y_set = dataset.get_set(set_type)

    if set_type == 'train':
        first_index, last_index = train_start, train_end
    elif set_type == 'test':
        first_index, last_index = test_start, test_end

    # Use Image Parameters
    img_rows, img_cols, nchannels = x_set.shape[1:4]
    nb_classes = y_set.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))

    with sess.as_default():
        model = load(filepath)
    assert len(model.get_params()) > 0

    # Initialize the Jacobian Saliency Map Method (JSMA) attack object and
    # graph
    jsma = SaliencyMapMethod(model, sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.3,
                 'clip_min': 0., 'clip_max': 1.,
                 'y_target': None, 'weighted': weighted}
    preds = model(x)
    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in range(first_index, last_index):
        results = pd.DataFrame()
        print('--------------------------------------')
        print('Attacking input %i/%i' % (sample_ind + 1, last_index))
        sample = x_set[sample_ind:(sample_ind + 1)]

        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(y_set[sample_ind]))
        target_classes = other_classes(nb_classes, current_class)

        # Loop over all target classes
        for target in target_classes:

            # This call runs the Jacobian-based saliency map approach
            one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
            one_hot_target[0, target] = 1
            jsma_params['y_target'] = one_hot_target
            adv_x, predictions = jsma.generate_np(sample, **jsma_params)

            # Check if success was achieved
            res = int(model_argmax(sess, x, preds, adv_x) == target)

            # Compute number of modified features
            adv_x_reshape = adv_x.reshape(-1)
            test_in_reshape = x_set[sample_ind].reshape(-1)
            nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
            percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]
            results['number_' + str(sample_ind) + '_' + str(current_class) + '_to_' + str(target)] = \
            np.concatenate((adv_x_reshape.reshape(-1),
                            predictions.reshape(-1),
                            np.array([nb_changed, percent_perturb, res])))

        sample_vector = sample.reshape(-1)
        shape1 = sample_vector.shape[0]
        shape2 = results.shape[0]

        results['original_image_' + str(sample_ind)] = np.concatenate((sample.reshape(-1), np.zeros((shape2-shape1,))))

        if weighted:
            attack_type = 'weighted'
        else:
            attack_type = 'simple'
        results.to_csv(save_path + '/' + attack_type + '_image_' + str(sample_ind) + '.csv', index = False)



