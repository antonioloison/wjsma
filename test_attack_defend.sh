echo "Testing jsma attack on the train dataset on model trained with jsma samples"
python start.py --job attack --dataset mnist-defense-jsma --settype train --weighted false --firstindex 0 --lastindex 2
echo "Testing wjsma attack on the train dataset on model trained with jsma samples"
python start.py --job attack --dataset mnist-defense-jsma --settype train --weighted true --firstindex 0 --lastindex 2
echo "Testing jsma attack on the test dataset on model trained with jsma samples"
python start.py --job attack --dataset mnist-defense-jsma --settype test --weighted false --firstindex 0 --lastindex 2
echo "Testing wjsma attack on the test dataset on model trained with jsma samples"
python start.py --job attack --dataset mnist-defense-jsma --settype test --weighted true --firstindex 0 --lastindex 2
echo "Testing jsma attack on the train dataset on model trained with wjsma samples"
python start.py --job attack --dataset mnist-defense-wjsma --settype train --weighted false --firstindex 0 --lastindex 2
echo "Testing wjsma attack on the train dataset on model trained with wjsma samples"
python start.py --job attack --dataset mnist-defense-wjsma --settype train --weighted true --firstindex 0 --lastindex 2
echo "Testing jsma attack on the test dataset on model trained with wjsma samples"
python start.py --job attack --dataset mnist-defense-wjsma --settype test --weighted false --firstindex 0 --lastindex 2
echo "Testing wjsma attack on the test dataset on model trained with wjsma samples"
python start.py --job attack --dataset mnist-defense-wjsma --settype test --weighted true --firstindex 0 --lastindex 2
read -p "Press enter to continue"
