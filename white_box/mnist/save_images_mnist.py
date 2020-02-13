"""
Generate images for the MNIST dataset
"""

import argparse

from cleverhans.dataset import MNIST

from white_box.generate_attacks import generate_attacks


MODEL_PATH = '../../models/joblibs/le_net_cleverhans_model.joblib'


def mnist_save_attacks(weighted, set_type, first_index, last_index):
    if weighted:
        attack_type = 'weighted'
    else:
        attack_type = 'simple'
    if set_type == 'test':
        generate_attacks('mnist_' + attack_type + '_' + set_type,
                     MODEL_PATH,
                     'test',
                     MNIST,
                     weighted,
                     test_start=first_index,
                     test_end=last_index)
    elif set_type == 'train':
        generate_attacks('mnist_' + attack_type + '_' + set_type,
                     MODEL_PATH,
                     set_type,
                     MNIST,
                     weighted,
                     train_start=first_index,
                     train_end=last_index)
    else:
        raise('Invalid set type')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weighted', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='test')
    parser.add_argument('--first_index', type=int, default=12)
    parser.add_argument('--last_index', type=float, default=13)
    args = parser.parse_args()

    mnist_save_attacks(args.weighted, args.dataset, args.first_index, args.last_index)
