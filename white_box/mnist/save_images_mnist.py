"""
Generate images for the MNIST dataset
"""

from cleverhans.dataset import MNIST

from white_box.generate_attacks import generate_attacks


MODEL_PATH = '../../models/joblibs/le_net_cleverhans_model.joblib'

def main():
    mnist_save_attacks(True, 'test', first_index=0, last_index=100)

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
                     True,
                     test_start=first_index,
                     test_end=last_index)
    elif set_type == 'train':
        generate_attacks('mnist_' + attack_type + '_' + set_type,
                     MODEL_PATH,
                     set_type,
                     MNIST,
                     True,
                     train_start=first_index,
                     train_end=last_index)
    else:
        raise('Invalid set type')

if __name__ == "__main__":
    main()
