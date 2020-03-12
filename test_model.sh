echo "Testing mnist model training"
python start.py --job train --dataset cifar10
echo "Testing cifar10 model training"
python start.py --job train --dataset cifar10
echo "Testing mnist model testing"
python start.py --job test --dataset mnist
echo "Testing cifar10 model testing"
python start.py --job test --dataset cifar10
read -p "Press enter to continue"
