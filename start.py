"""
Arguments
-- job [train, test, attack, augment, stats, visualisation]
-- dataset [mnist, cifar10, mnist-defense-jsma, mnist-defense-wjsma]
-- settype [test, train]
-- weighted [false, true]
-- firstindex int
-- lastindex int
-- visual [probabilities, single, line, square]

Available jobs (see README.md for extra information)
train (dataset)
test (dataset)
attack (dataset, settype, weighted, firstindex, lastindex)
augment (settype, weighted)
stats (dataset, settype, weighted)
visualisation (visual)
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=str, default="attack")
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--settype', type=str, default="test")
    parser.add_argument('--weighted', type=str, default="true")
    parser.add_argument('--firstindex', type=int, default=0)
    parser.add_argument('--lastindex', type=int, default=1)
    parser.add_argument('--visual', type=str, default='single')
    args = parser.parse_args()

    if args.job == "train":
        if args.dataset == "mnist":
            from models.mnist import model_train

            model_train()
        elif args.dataset == "cifar10":
            from models.cifar10 import model_train

            model_train()
        elif args.dataset == "mnist-defense-jsma":
            from defense.train_defense import model_train

            model_train(False)
        elif args.dataset == "mnist-defense-wjsma":
            from defense.train_defense import model_train

            model_train(True)
        else:
            raise ValueError("Invalid dataset")
    elif args.job == "test":
        if args.dataset == "mnist":
            from models.mnist import model_test

            model_test()
        elif args.dataset == "cifar10":
            from models.cifar10 import model_test

            model_test()
        elif args.dataset == "mnist-defense-jsma":
            from defense.train_defense import model_test

            model_test(False)
        elif args.dataset == "mnist-defense-wjsma":
            from defense.train_defense import model_test

            model_test(True)
        else:
            raise ValueError("Invalid dataset")
    elif args.job == "attack":
        if args.settype != "test" and args.settype != "train":
            raise ValueError("Invalid set type")

        if args.weighted != "true" and args.weighted != "false":
            raise ValueError("wjsma argument is invalid")

        if args.dataset == "mnist":
            from attack.mnist.save_images_mnist import mnist_save_attacks

            mnist_save_attacks(args.weighted == "true", args.settype, args.firstindex, args.lastindex)
        elif args.dataset == "cifar10":
            from attack.cifar10.save_images_cifar10 import cifar10_save_attacks

            cifar10_save_attacks(args.weighted == "true", args.settype, args.firstindex, args.lastindex)
        elif args.dataset == "mnist-defense-jsma":
            from defense.mnist_defense_jsma.save_images_mnist_defense import mnist_defense_save_attacks

            mnist_defense_save_attacks(args.weighted == "true", args.settype, args.firstindex, args.lastindex)
        elif args.dataset == "mnist-defense-wjsma":
            from defense.mnist_defense_wjsma.save_images_mnist_defense import mnist_defense_save_attacks

            mnist_defense_save_attacks(args.weighted == "true", args.settype, args.firstindex, args.lastindex)
        else:
            raise ValueError("Invalid dataset")
    elif args.job == "augment":
        if args.settype != "test" and args.settype != "train":
            raise ValueError("Invalid set type")

        if args.weighted != "true" and args.weighted != "false":
            raise ValueError("wjsma argument is invalid")

        from defense.sample_selection import generate_extra_set

        generate_extra_set(args.settype, args.weighted == "true")
    elif args.job == "stats":
        if args.settype != "test" and args.settype != "train":
            raise ValueError("Invalid set type")

        if args.weighted != "true" and args.weighted != "false":
            raise ValueError("wjsma argument is invalid")

        from stats.stats import average_stat

        if args.dataset == "mnist":
            if args.settype == "test":
                if args.weighted == "false":
                    average_stat("attack/mnist/jsma_test/")
                else:
                    average_stat("attack/mnist/wjsma_test/")
            else:
                if args.weighted == "false":
                    average_stat("attack/mnist/jsma_train/")
                else:
                    average_stat("attack/mnist/wjsma_train/")
        elif args.dataset == "cifar10":
            if args.settype == "test":
                if args.weighted == "false":
                    average_stat("attack/cifar10/jsma_test/")
                else:
                    average_stat("attack/cifar10/wjsma_test/")
            else:
                if args.weighted == "false":
                    average_stat("attack/cifar10/jsma_train/")
                else:
                    average_stat("attack/cifar10/wjsma_train/")
        elif args.dataset == "mnist-defense-jsma":
            if args.settype == "test":
                if args.weighted == "false":
                    average_stat("defense/mnist_defense_jsma/jsma_test/")
                else:
                    average_stat("defense/mnist_defense_jsma/wjsma_test/")
            else:
                if args.weighted == "false":
                    average_stat("defense/mnist_defense_jsma/jsma_train/")
                else:
                    average_stat("defense/mnist_defense_jsma/wjsma_train/")
        elif args.dataset == "mnist-defense-wjsma":
            if args.settype == "test":
                if args.weighted == "false":
                    average_stat("defense/mnist_defense_wjsma/jsma_test/")
                else:
                    average_stat("defense/mnist_defense_wjsma/wjsma_test/")
            else:
                if args.weighted == "false":
                    average_stat("defense/mnist_defense_wjsma/jsma_train/")
                else:
                    average_stat("defense/mnist_defense_wjsma/wjsma_train/")
        else:
            raise ValueError("Invalid dataset")
    elif args.job == "visualisation":
        if args.visual == "probabilities":
            from visualisation.show_probabilities import visualise

            visualise(r"attack/mnist/")
        elif args.visual not in ["single", "line", "square"]:
            raise ValueError("Invalid visualisation mode")
        else:
            if args.visual == "single":
                from visualisation.show_image import single_image

                single_image(r"attack/cifar10", 5, 8)
            elif args.visual == "line":
                from visualisation.show_image import one_line

                one_line(r"attack/mnist/wjsma_test/wjsma_image_5.csv")
            elif args.visual == "square":
                from visualisation.show_image import image_square

                image_square(r"attack/cifar10/")
    else:
        raise ValueError("Invalid job")
