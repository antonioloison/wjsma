echo "Testing stats of jsma samples from cifar10 train dataset"
python start.py --job stats --dataset cifar10 --settype train --weighted false
echo "Testing stats of wjsma samples from cifar10 train dataset"
python start.py --job stats --dataset cifar10 --settype train --weighted true
echo "Testing stats of jsma samples from cifar10 test dataset"
python start.py --job stats --dataset cifar10 --settype test --weighted false
echo "Testing stats of wjsma samples from cifar10 test dataset"
python start.py --job stats --dataset cifar10 --settype test --weighted true
echo "Testing stats of jsma samples from mnist train dataset"
python start.py --job stats --dataset mnist --settype train --weighted false
echo "Testing stats of wjsma samples from mnist train dataset"
python start.py --job stats --dataset mnist --settype train --weighted true
echo "Testing stats of jsma samples from mnist test dataset"
python start.py --job stats --dataset mnist --settype test --weighted false
echo "Testing stats of wjsma samples from mnist test dataset"
python start.py --job stats --dataset mnist --settype test --weighted true
echo "Testing stats of jsma samples on model trained with jsma samples from mnist train dataset"
python start.py --job stats --dataset mnist-defense-simple --settype train --weighted false
echo "Testing stats of wjsma samples on model trained with jsma samples from mnist train dataset"
python start.py --job stats --dataset mnist-defense-simple --settype train --weighted true
echo "Testing stats of jsma samples on model trained with jsma samples from mnist test dataset"
python start.py --job stats --dataset mnist-defense-simple --settype test --weighted false
echo "Testing stats of wjsma samples on model trained with jsma samples from mnist test dataset"
python start.py --job stats --dataset mnist-defense-simple --settype test --weighted true
echo "Testing stats of jsma samples on model trained with wjsma samples from mnist train dataset"
python start.py --job stats --dataset mnist-defense-weighted --settype train --weighted false
echo "Testing stats of wjsma samples on model trained with wjsma samples from mnist train dataset"
python start.py --job stats --dataset mnist-defense-weighted --settype train --weighted true
echo "Testing stats of jsma samples on model trained with wjsma samples from mnist test dataset"
python start.py --job stats --dataset mnist-defense-weighted --settype test --weighted false
echo "Testing stats of wjsma samples on model trained with wjsma samples from mnist test dataset"
python start.py --job stats --dataset mnist-defense-weighted --settype test --weighted true
read -p "Press enter to continue"
