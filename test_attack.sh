echo "Testing jsma attack on cifar10 images from the train dataset"
python start.py --job attack --dataset cifar10 --settype train --weighted false --firstindex 0 --lastindex 2
echo "Testing wjsma attack on cifar10 images from the train dataset"
python start.py --job attack --dataset cifar10 --settype train --weighted true --firstindex 0 --lastindex 2
echo "Testing jsma attack on cifar10 images from the test dataset"
python start.py --job attack --dataset cifar10 --settype test --weighted false --firstindex 0 --lastindex 2
echo "Testing wjsma attack on cifar10 images from the test dataset"
python start.py --job attack --dataset cifar10 --settype test --weighted true --firstindex 0 --lastindex 2
echo "Testing jsma attack on mnist images from the train dataset"
python start.py --job attack --dataset mnist --settype train --weighted false --firstindex 10 --lastindex 2
echo "Testing wjsma attack on mnist images from the train dataset"
python start.py --job attack --dataset mnist --settype train --weighted true --firstindex 10 --lastindex 2
echo "Testing jsma attack on mnist images from the test dataset"
python start.py --job attack --dataset mnist --settype test --weighted false --firstindex 10 --lastindex 100
echo "Testing wjsma attack on mnist images from the test dataset"
python start.py --job attack --dataset mnist --settype test --weighted true --firstindex 10 --lastindex 100
read -p "Press enter to continue"
