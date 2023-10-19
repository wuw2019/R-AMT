#!/bin/bash

# custom config
DATA=/usr/zkc/data
TRAINER=CoCoOp

DATASET=imagenet
CFG=rn50_b16_c4_ep10_batch1 # config file

SHOTS=16  # number of shots (1, 2, 4, 8, 16)
SUB=base
LOADEP=10

for SEED in 1 #2 3
do
    DIR=output/evaluation/${TRAINER}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}/epoch${LOADEP}_${SUB}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=0,1 python train.py \
    	--root ${DATA} \
    	--seed ${SEED} \
    	--trainer ${TRAINER} \
    	--dataset-config-file configs/datasets/${DATASET}.yaml \
    	--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    	--output-dir ${DIR} \
	--model-dir output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED} \
    	--eval-only \
	--load-epoch ${LOADEP} \
	DATASET.NUM_SHOTS ${SHOTS} \
    	DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
