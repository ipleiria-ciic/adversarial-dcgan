#!/bin/sh

ATTACK=FGSM
MODEL=AlexNet
EPOCHS=60
BATCH=128
EPOCH_TO_RESUME=0
DELTAS="0.01 0.05 0.10 0.15 0.20"

for DELTA in $DELTAS; do
    # 1. Attacks (Perturbation)
    python Attacks/$ATTACK.py --model $MODEL --delta $DELTA

    # 2. DCGAN
    python DCGAN/main.py --attack $ATTACK --epochs $EPOCHS --batch $BATCH --resume $EPOCH_TO_RESUME --delta $DELTA

    # 3. Export
    # python DCGAN/export-images.py --attack $ATTACK --epochs $EPOCHS --batch $BATCH

    # 4. Evaluation
    # python Evaluation/Evaluation.py --attack $ATTACK --model $MODEL
done