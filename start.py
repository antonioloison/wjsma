"""
Arguments
-- job [train, test, attack, augment, stats, visualisation]
-- dataset [mnist, cifar10, mnist_defense_jsma, mnist_defense_wjsma, mnist_defense_tjsma]
-- settype [test, train]
-- attack [jsma, wjsma, tjsma]
-- firstindex int
-- lastindex int
-- visual [probabilities, single, line, square]

Available jobs (see README.md for extra information)
train (dataset)
test (dataset)
attack (dataset, settype, attack, firstindex, lastindex)
augment (settype, attack)
stats (dataset, settype, attack)
visualisation (visual)
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=str, default="attack")
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--settype', type=str, default="test")
    parser.add_argument('--attack', type=str, default="jsma")
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
        elif args.dataset == "mnist_defense_jsma":
            from defense.train_mnist_defense import model_train

            model_train("jsma")
        elif args.dataset == "mnist_defense_wjsma":
            from defense.train_mnist_defense import model_train

            model_train("wjsma")
        elif args.dataset == "mnist_defense_tjsma":
            from defense.train_mnist_defense import model_train

            model_train("tjsma")
        else:
            raise ValueError("Invalid dataset")
    elif args.job == "test":
        if args.dataset == "mnist":
            from models.mnist import model_test

            model_test()
        elif args.dataset == "cifar10":
            from models.cifar10 import model_test

            model_test()
        elif args.dataset == "mnist_defense_jsma":
            from defense.train_mnist_defense import model_test

            model_test("jsma")
        elif args.dataset == "mnist_defense_wjsma":
            from defense.train_mnist_defense import model_test

            model_test("wjsma")
        elif args.dataset == "mnist_defense_tjsma":
            from defense.train_mnist_defense import model_test

            model_test("tjsma")
        else:
            raise ValueError("Invalid dataset")
    elif args.job == "attack":
        if args.settype != "test" and args.settype != "train":
            raise ValueError("Invalid set type")

        if args.attack != "jsma" and args.attack != "wjsma" and args.attack != "tjsma":
            raise ValueError("attack argument is invalid")
        
        from attack.save_images import save_images
        
        save_images(args.dataset, args.attack, args.settype, args.firstindex, args.lastindex)
    elif args.job == "augment":
        if args.settype != "test" and args.settype != "train":
            raise ValueError("Invalid set type")

        if args.attack != "jsma" and args.attack != "wjsma" and args.attack != "tjsma":
            raise ValueError("attack argument is invalid")

        from defense.sample_selection import generate_extra_set

        generate_extra_set(args.settype, args.attack)
    elif args.job == "stats":
        if args.settype != "test" and args.settype != "train":
            raise ValueError("Invalid set type")

        if args.attack != "jsma" and args.attack != "wjsma" and args.attack != "tjsma":
            raise ValueError("attack argument is invalid")

        from stats.stats import average_stat

        average_stat(args.dataset, args.settype, args.attack)
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
