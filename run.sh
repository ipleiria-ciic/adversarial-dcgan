#!/bin/sh

ATTACK=FGSM
MODEL=AlexNet
EPOCHS=10000
BATCH=64
EPOCH_TO_RESUME=5000

# Workflow Order: Attacks -> DCGAN -> Export -> Evaluation

# 1. Attacks (Perturbation)
python Attacks/$ATTACK.py --model $MODEL

# 2. DCGAN
python DCGAN/DCGAN.py --attack $ATTACK --epochs $EPOCHS --batch $BATCH --resume $EPOCH_TO_RESUME

# 3. Export
python DCGAN/Export.py --attack $ATTACK --epochs $EPOCHS --batch $BATCH

# 4. Evaluation
python Evaluation/Evaluation.py --attack $ATTACK --model $MODEL
