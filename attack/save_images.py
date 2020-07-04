from attack.generate_attacks import generate_attacks


MNIST_SETS = ["mnist", "mnist_defense_jsma", "mnist_defense_wjsma", "mnist_defense_tsma"]
CIFAR10_SETS = ["cifar10", "cifar10_defense_jsma", "cifar10_defense_wjsma", "cifar10_defense_tsma"]


def save_images(model, attack, set_type, first_index, last_index):
    if model in MNIST_SETS:
        from cleverhans.dataset import MNIST

        x_set, y_set = MNIST(train_start=0, train_end=60000, test_start=0, test_end=10000).get_set(set_type)
    elif model in CIFAR10_SETS:
        from cleverhans.dataset import CIFAR10

        x_set, y_set = CIFAR10(train_start=0, train_end=50000, test_start=0, test_end=0).get_set(set_type)
        y_set = y_set.reshape((y_set.shape[0], 10))
    else:
        raise ValueError("Invalid model: " + model)

    generate_attacks(
        save_path="attack/" + model + "/" + attack + "_" + set_type,
        file_path="models/joblibs/" + model + ".joblib",
        x_set=x_set,
        y_set=y_set,
        attack=attack,
        first_index=first_index,
        last_index=last_index
    )
