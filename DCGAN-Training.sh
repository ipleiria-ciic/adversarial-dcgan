#!/bin/sh

ATTACK=TRM
MODEL=AlexNet
EPOCHS=50
BATCH=128
EPOCH_TO_RESUME=0
DELTAS="0.01 0.05 0.10 0.15 0.20"

for DELTA in $DELTAS; do
    # 1. Attacks (Perturbation)
    python Attacks/$ATTACK.py --model $MODEL --delta $DELTA

    # 2. DCGAN
    python DCGAN/DCGAN.py --attack $ATTACK --epochs $EPOCHS --batch $BATCH --resume $EPOCH_TO_RESUME --delta $DELTA
done