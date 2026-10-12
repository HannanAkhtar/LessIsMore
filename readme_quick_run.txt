Run whole-frame LR-wide:

python run_framework.py wholeframe --data_root ../data --only bunny honeybee jockey readysetgo yachtride --epochs 300 --gpu 0

Run 4-tile spatial:

python run_framework.py spatial4 --data_root ../data --sequence bunny --epochs 300 --gpu 0 --clean

Run 16-tile spatial:

python run_framework.py spatial16 --data_root ../data --sequence bunny --epochs 300 --gpu 0 --clean