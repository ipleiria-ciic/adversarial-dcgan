#!/bin/bash

ATTACK=TRM
BATCH=128

DELTAS=(0.01 0.05 0.10 0.15 0.20)
BEST_CHECKPOINTS=(24 35 6 28 27)

for i in ${!DELTAS[@]}; do
    DELTA=${DELTAS[$i]}
    CHECKPOINT=${BEST_CHECKPOINTS[$i]}
    python Testing/Testing.py --attack $ATTACK --checkpoint $CHECKPOINT --delta $DELTA
done