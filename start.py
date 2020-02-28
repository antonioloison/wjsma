"""
Fusion entry point
-- job [train, test, attack, augment]
-- dataset [mnist, cifar10, mnist-defense-simple, mnist-defense-weighted]
-- settype [test, train]
-- weighted [False, True]
-- firstindex int
-- lastindex int

train (dataset)
test (dataset)
attack (settype, dataset, weighted, firstindex, lastindex)
augment (weighted)
"""


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--job', type=str, default="train")
parser.add_argument('--dataset', type=str, default="mnist")
parser.add_argument('--settype', type=str, default="test")
parser.add_argument('--weighted', type=str, default="false")
parser.add_argument('--firstindex', type=int, default=0)
parser.add_argument('--lastindex', type=int, default=10000)
args = parser.parse_args()

if args.job == "train":
    if args.dataset == "mnist":
        from models.mnist_clean import model_train

        model_train()
    elif args.dataset == "cifar10":
        from models.cifar10 import model_train

        model_train()
    elif args.dataset == "mnist-defense-simple":
        from defense.train_defense import model_train

        model_train(False)
    elif args.dataset == "mnist-defense-weighted":
        from defense.train_defense import model_train

        model_train(True)
    else:
        raise ValueError("Invalid dataset")
elif args.job == "test":
    if args.dataset == "mnist":
        from models.mnist_clean import model_test

        model_test()
    elif args.dataset == "cifar10":
        from models.cifar10 import model_test

        model_test()
    elif args.dataset == "mnist-defense-simple":
        from defense.train_defense import model_test

        model_test(False)
    elif args.dataset == "mnist-defense-weighted":
        from defense.train_defense import model_test

        model_test(True)
    else:
        raise ValueError("Invalid dataset")
elif args.job == "attack":
    if args.settype != "test" and args.settype != "train":
        raise ValueError("Invalid set type")

    if args.weighted != "true" and args.weighted != "false":
        raise ValueError("Weighted argument is invalid")

    if args.dataset == "mnist":
        from white_box.mnist.save_images_mnist import mnist_save_attacks

        mnist_save_attacks(args.weighted == "true", args.settype, args.firstindex, args.lastindex)
    elif args.dataset == "cifar10":
        from white_box.cifar10.save_images_cifar10 import cifar10_save_attacks

        cifar10_save_attacks(args.weighted == "true", args.settype, args.firstindex, args.lastindex)
    elif args.dataset == "mnist-defense-simple":
        from defense.mnist_defense_simple.save_images_mnist_defense import mnist_defense_save_attacks

        mnist_defense_save_attacks(args.weighted == "true", args.settype, args.firstindex, args.lastindex)
    elif args.dataset == "mnist-defense-weighted":
        from defense.mnist_defense_weighted.save_images_mnist_defense import mnist_defense_save_attacks

        mnist_defense_save_attacks(args.weighted == "true", args.settype, args.firstindex, args.lastindex)
    else:
        raise ValueError("Invalid dataset")
elif args.job == "augment":
    if args.weighted != "true" and args.weighted != "false":
        raise ValueError("Weighted argument is invalid")

    from defense.sample_selection import generate_extra_set

    generate_extra_set(args.weighted == "true")
elif args.job == "stats":
    if args.settype != "test" and args.settype != "train":
        raise ValueError("Invalid set type")

    if args.weighted != "true" and args.weighted != "false":
        raise ValueError("Weighted argument is invalid")

    from stats.stats import average_stat

    if args.dataset == "mnist":
        if args.settype == "test":
            if args.weighted == "false":
                average_stat("white_box/mnist/simple_test/")
            else:
                average_stat("white_box/mnist/weighted_test/")
        else:
            if args.weighted == "false":
                average_stat("white_box/mnist/simple_train/")
            else:
                average_stat("white_box/mnist/weighted_train/")
    elif args.dataset == "cifar10":
        if args.settype == "test":
            if args.weighted == "false":
                average_stat("white_box/cifar10/simple_test/")
            else:
                average_stat("white_box/cifar10/weighted_test/")
        else:
            if args.weighted == "false":
                average_stat("white_box/cifar10/simple_train/")
            else:
                average_stat("white_box/cifar10/weighted_train/")
    elif args.dataset == "mnist-defense-simple":
        if args.settype == "test":
            if args.weighted == "false":
                average_stat("defense/mnist_defense_simple/simple_test/")
            else:
                average_stat("defense/mnist_defense_simple/weighted_test/")
        else:
            if args.weighted == "false":
                average_stat("defense/mnist_defense_simple/simple_train/")
            else:
                average_stat("defense/mnist_defense_simple/weighted_train/")
    elif args.dataset == "mnist-defense-weighted":
        if args.settype == "test":
            if args.weighted == "false":
                average_stat("defense/mnist_defense_weighted/simple_test/")
            else:
                average_stat("defense/mnist_defense_weighted/weighted_test/")
        else:
            if args.weighted == "false":
                average_stat("defense/mnist_defense_weighted/simple_train/")
            else:
                average_stat("defense/mnist_defense_weighted/weighted_train/")
    else:
        raise ValueError("Invalid dataset")
else:
    raise ValueError("Invalid job")
