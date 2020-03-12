echo "Testing jsma attack on the train dataset on model trained with jsma samples"
python start.py --job attack --dataset mnist-defense-simple --settype train --weighted false --firstindex 0 --lastindex 2
echo "Testing wjsma attack on the train dataset on model trained with jsma samples"
python start.py --job attack --dataset mnist-defense-simple --settype train --weighted true --firstindex 0 --lastindex 2
echo "Testing jsma attack on the test dataset on model trained with jsma samples"
python start.py --job attack --dataset mnist-defense-simple --settype test --weighted false --firstindex 0 --lastindex 2
echo "Testing wjsma attack on the test dataset on model trained with jsma samples"
python start.py --job attack --dataset mnist-defense-simple --settype test --weighted true --firstindex 0 --lastindex 2
echo "Testing jsma attack on the train dataset on model trained with wjsma samples"
python start.py --job attack --dataset mnist-defense-weighted --settype train --weighted false --firstindex 0 --lastindex 2
echo "Testing wjsma attack on the train dataset on model trained with wjsma samples"
python start.py --job attack --dataset mnist-defense-weighted --settype train --weighted true --firstindex 0 --lastindex 2
echo "Testing jsma attack on the test dataset on model trained with wjsma samples"
python start.py --job attack --dataset mnist-defense-weighted --settype test --weighted false --firstindex 0 --lastindex 2
echo "Testing wjsma attack on the test dataset on model trained with wjsma samples"
python start.py --job attack --dataset mnist-defense-weighted --settype test --weighted true --firstindex 0 --lastindex 2
read -p "Press enter to continue"
