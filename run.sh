#!/usr/bin/env bash

export TRAINING_DATA=data/train_folds.csv
export TEST_DATA=data/test.csv
#export FOLD=$2
export MODEL=$1

#FOLD=0 python -m src.train
#FOLD=1 python -m src.train
#FOLD=2 python -m src.train
#FOLD=3 python -m src.train
#FOLD=4 python -m src.train

python -m src.predict