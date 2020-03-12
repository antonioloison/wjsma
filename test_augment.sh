echo "Testing dataset augmentation with jsma samples"
python start.py --job augment --settype test --weighted false
echo "Testing dataset augmentation with wjsma samples"
python start.py --job augment --settype test --weighted true
read -p "Press enter to continue"
