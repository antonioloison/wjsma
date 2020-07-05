# An Efficient Probabilistic Jacobian-based Saliency Map Attack in Machine Learning (WJSMA)

## Installation

Please install the packages required by the `requirements.txt` file.
In case you encounter some errors after this installation, you can refer to section `Model Training and Testing Precautions` to fix them.

## How to use

For reproduction, you can use the script `start.py` to run every task used in the paper

`python start.py --job <job> --dataset <dataset> --settype <settype> --attack <attack> --firstindex <firstindex> --lastindex <lastindex> --visual <visual>`
- `job` either `"train"`, `"test"`, `"attack"`, `"augment"`, `"stats"` or `"visualisation"`, selects the action that the script will run (see below for examples)
- `dataset` either `"mnist"`, `"cifar10"`, `"mnist_defense_jsma"`, `"mnist_defense_wjsma"`, `"mnist_defense_tjsma"`, `"cifar10_defense_jsma"`, `"cifar10_defense_wjsma"` or `"cifar10_defense_tjsma"`, selects the dataset / model on which the job will be performed (note that `"mnist_defense_jsma"`, `"mnist_defense_wjsma"`, `"mnist_defense_tjsma"`, `"cifar10_defense_jsma"`, `"cifar10_defense_wjsma"` and `"cifar10_defense_tjsma"` are trained on the augmented version of the MNIST and CIFAR10 dataset)
- `settype` either `"train"` or `"test"`, switches between the train and the test of the dataset
- `attack` either `"jsma"`, `"wjsma"` or `"tjsma"`, switches between Papernot's JSMA and our implementation of WJSMA and TJSMA
- `firstindex` an integer (only used for the attack, specifies the index of the first attacked image in the dataset)
- `lastindex` an integer (only used for the attack, specifies the index of the last attacked image in the dataset)
- `visual` either `"probabilities"`, `"single"`, `"line"`, `"square"`, switches between the type of image visualisation

### Job examples

#### Models

To create a new LeNet5 model and train it on the original MNIST dataset

`python start.py --job train --dataset mnist`

To test an existing LeNet5 model trained over the original MNIST dataset

`python start.py --job test --dataset mnist`

#### Adversarial examples generation

To generate WJSMA adversarial samples against the previously trained LeNet5 model over the train set of the MNIST dataset

`python start.py --job attack --dataset mnist --settype train --attack wjsma --firstindex 0 --lastindex 10000`

To generate TJSMA adversarial samples against the previously trained LeNet5 model over the train set of the MNIST dataset

`python start.py --job attack --dataset mnist --settype train --attack tjsma --firstindex 0 --lastindex 10000`

#### Defenses

To generate the augmented MNIST dataset using the previously crafted adversarial samples (note that you can only augment the original MNIST dataset)

`python start.py --job augment --settype train --attack wjsma`

To train a new LeNet5 model and train it on the augmented MNIST dataset

`python start.py --job train --dataset mnist_defense_wjsma`

To generate WJSMA adversarial samples against the newly trained LeNet5 model over the test set of the MNIST dataset

`python start.py --job attack --dataset mnist_defense_wjsma --settype test --attack wjsma --firstindex 0 -- lastindex 10000`

#### Analyse attack and model performances

To print out the performances of the different attacks

`python start.py --job stats --dataset mnist_defense_wjsma --settype test --attack wjsma`

#### Visualise images

To show and compare adversarial samples

`python start.py --job visualisation --visual single`

### CSV File Structure of the Adversarial Samples

Each csv file has ten columns. The first nine columns contain the adversarial samples for each target different from the origin class, while the last column contains the original image.
In each adversarial sample column, the first (784 for MNIST images and 3072 for CIFAR-10 images) lines contain the pixel values of the adversarial samples, the last three lines contain the number of changed pixels, the distortion coefficient and if the attack was successful.

### Model Training and Testing Precautions

The joblib files in the `joblib` file are the models that we used for our simulations. If you try to train a new neural networks, these models will be overwritten. To avoid that, you only need to rename our original models.

#### Loading errors handling

When loading a model for attacks or testing, you may encounter the following errors: 
- `AttributeError: module 'cleverhans.picklable_model' has no attribute 'MaxPooling2D'`
- `ModuleNotFoundError: No module named 'cleverhans_utils'` 

To solve the `AttributeError`, you can copy the MaxPooling2D layer in the `cleverhans_utils.py` file of the `models` folder. Then, paste it in `picklable_model.py` of the cleverhans library code under the `GlobalAveragePool(Layer)`.
Then delete the import `from models.cleverhans_utils import MaxPooling2D` and add `MaxPooling2D` to this import `from cleverhans.picklable_model import Conv2D, ReLU, Softmax, MLP, GlobalAveragePool`

To solve the `ModuleNotFoundError`, add the following lines to the top of the file `start.py`:

```
# Replace YOUR_MODEL_PATH by the path of the models folder of the form /Users/user/.../wjsma/models
import sys
sys.path.append(YOUR_MODEL_PATH)
```
