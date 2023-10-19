#!/bin/bash

# custom config
DATA=../../datasets/
TRAINER=ZeroshotCLIP
DATASET=imagenet_a
CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/MASK/imagenet/${CFG}.yaml \
--output-dir ../output/${TRAINER}/${CFG}/${DATASET} \
--eval-only
