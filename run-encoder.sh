#!/bin/sh

ATTACK=FGSM
EPOCHS=5000
BATCH=64

python DCGAN/encoder-training.py --attack $ATTACK --epochs $EPOCHS --batch $BATCH