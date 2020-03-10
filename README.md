# An Efficient Probabilistic Jacobian-based Saliency Map Attack in Machine Learning (WJSMA)

## How to use

For reproduction, you can use the script `start.py` to run every tasks used in the paper

`python start.py --job <job> --dataset <dataset> --settype <settype> --weighted <weighted> --firstindex <firstindex> --lastindex <lastindex>`
- `job` either `"train"`, `"test"`, `"attack"`, `"augment"` or `"stats"`, selects the action that the script will run (see below for examples)
- `dataset` either `"mnist"`, `"cifar10"`, `"mnist-defense-simple"` or `"mnist-defense-weighted"`, selects the dataset / model on which the job will be performed (note that `mnist-defense-simple` and `mnist-defense-weighted` are the MNIST datasets augmented by JSMA and WJSMA respectively)
- `settype` either `"train"` or `"test"`, switches between the train and the test of the dataset
- `weighted` either `"true"` or `"false"`, switches between Papernot's JSMA and WJSMA
- `firstindex` an integer (only used for the attack, specifies the index of the first attacked image in the dataset)
- `lastindex` an integer (only used for the attack, specifies the index of the last attacked image in the dataset)

### Job examples

To create a new LeNet5 model and train it on the original MNIST dataset

`python start.py --job train --dataset mnist`

To test an existing LeNet5 model trained over the original MNIST dataset

`python start.py --job test --dataset mnist`

To generate WJSMA adversarial samples against the previously trained LeNet5 model over the train set of the MNIST dataset

`python start.py --job attack --dataset mnist --settype train --weighted true --firstindex 0 --lastindex 10000`

To generate the augmented MNIST dataset using the previously crafted adversarial samples (note that you can only augment the original MNIST dataset)

`python start.py --job augment --settype train --weighted true`

To train a new LeNet5 model and train it on the augmented MNIST dataset

`python start.py --job train --dataset mnist-defense-weighted`

To generate WJSMA adversarial samples against the newly trained LeNet5 model over the test set of the MNIST dataset

`python start.py --job attack --dataset mnist-defense-weighted --settype test --weighted true --firstindex 0 -- lastindex 10000`

To print out the performances of our model previously attacked

`python start.py --job stats --datatset mnist-defense-weighted --settype test --weighted true`