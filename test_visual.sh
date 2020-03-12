echo "Testing probabilities visualisation"
python start.py --job visualisation --visual probabilities
echo "Testing single image visualisation"
python start.py --job visualisation --visual single
echo "Testing line visualisation"
python start.py --job visualisation --visual line
echo "Testing image square visualisation"
python start.py --job visualisation --visual square
echo "Testing visualisation with bad argument"
python start.py --job visualisation --visual badarg
read -p "Press enter to continue"
