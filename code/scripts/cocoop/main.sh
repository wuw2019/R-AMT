#!/bin/bash

# custom config
DATA=../../
TRAINER=CoCoOp

DATASET=imagenet
CFG=vit_b16_c4_ep10_batch1 # config file

SHOTS=16  # number of shots (1, 2, 4, 8, 16)

for SEED in 1 #2 3
do
    DIR=../output/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
    	--root ${DATA} \
    	--seed ${SEED} \
    	--trainer ${TRAINER} \
    	--dataset-config-file configs/datasets/${DATASET}.yaml \
    	--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    	--output-dir ${DIR} \
		DATALOADER.TRAIN_X.BATCH_SIZE 1 \
    	DATASET.NUM_SHOTS ${SHOTS} \
    	DATASET.SUBSAMPLE_CLASSES base
    fi
done
