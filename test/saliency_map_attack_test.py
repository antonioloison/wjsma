from saliency_map_attack import *

from cleverhans.serial import load
from cleverhans.dataset import MNIST
from cleverhans.utils import other_classes
from cleverhans.utils_tf import model_argmax

import tensorflow as tf


SAMPLE_IND = 1002
NB_CLASSES = 10

def saliency_map_attack_test(weighted):
    session = tf.Session()
    with session.as_default():
        model = load(r"C:\Users\Antonio\Projects\wjsma\models\joblibs\le_net_cleverhans_model.joblib")
    jsma_params = {'theta': 1., 'gamma': 0.3,
                        'clip_min': 0., 'clip_max': 1.,
                        'y_target': None, 'weighted': weighted}
    jsma_method = SaliencyMapMethod(model, sess=session)

    if weighted:
        attack_type = 'weighted'
    else:
        attack_type = 'simple'

    mnist = MNIST(train_start=0, train_end=60000,
                    test_start=0, test_end=10000)

    x_train, y_train = mnist.get_set('train')
    img_rows, img_cols, nchannels = x_train.shape[1:4]

    sample = x_train[SAMPLE_IND:(SAMPLE_IND+1)]

    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                            nchannels))
    preds = model.get_logits(x)

    # We want to find an adversarial example for each possible target class
    # (i.e. all classes that differ from the label given in the dataset)
    current_class = int(np.argmax(y_train[SAMPLE_IND]))
    target_classes = other_classes(NB_CLASSES, current_class)

    # Loop over all target classes
    for target in target_classes:
        print('Generating adv. example for target class %i' % target)

        # This call runs the Jacobian-based saliency map approach
        one_hot_target = np.zeros((1, NB_CLASSES), dtype=np.float32)
        one_hot_target[0, target] = 1
        jsma_params['y_target'] = one_hot_target
        adv_x = jsma_method.generate_np(sample, **jsma_params)
        # Check if success was achieved
        res = int(model_argmax(session, x, preds, adv_x) == target)
        image = np.load('test_images/image_' + str(SAMPLE_IND) + '_' + attack_type + '_' + str(target) + '.npy')
        assert(np.array_equal(image,adv_x))
        print('Generation went good')


saliency_map_attack_test('weighted')
