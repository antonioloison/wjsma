"""
Fusion entry point
-- job [train, test, attack, augment, defense]
-- dataset [mnist, cifar10]
-- settype [test, train]
-- weighted [False, True]
-- firstindex int
-- lastindex int

train (dataset)
test (dataset)
attack (settype, dataset, weighted, firstindex, lastindex)
augment (weighted)
defense ()
"""


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--job', type=str, default="train")
parser.add_argument('--dataset', type=str, default="mnist")
parser.add_argument('--settype', type=str, default="test")
parser.add_argument('--weighted', type=bool, default=False)
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
    else:
        raise ValueError("Invalid dataset")
elif args.job == "test":
    if args.dataset == "mnist":
        from models.mnist_clean import model_test

        model_test()
    elif args.dataset == "cifar10":
        from models.cifar10 import model_test

        model_test()
    else:
        raise ValueError("Invalid dataset")
elif args.job == "attack":
    if args.settype != "test" or args.settype != "train":
        raise ValueError("Invalid set type")

    if args.dataset == "mnist":
        from white_box.mnist.save_images_mnist import mnist_save_attacks

        mnist_save_attacks(args.weighted, args.settype, args.firstindex, args.lastindex)
    elif args.dataset == "cifar10":
        from white_box.cifar10.save_images_cifar10 import cifar10_save_attacks

        cifar10_save_attacks(args.weighted, args.settype, args.firstindex, args.lastindex)
elif args.job == "augment":
    from defense.sample_selection import generate_extra_set

    generate_extra_set(args.weighted)
elif args.job == "defense":
    raise NotImplementedError("Not implemented yet")
else:
    raise ValueError("Invalid job")
