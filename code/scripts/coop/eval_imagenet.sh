#!/bin/bash

# custom config
DATA=../../
TRAINER=CoOp

DATASET=imagenet
CFG=vit_b16 # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
# SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)


for SHOTS in 16 
do
for SEED in 1
do
    DIR=../output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}_ep50.yaml \
    --output-dir ${DIR}/eval \
    --eval-only \
    DATALOADER.TEST.BATCH_SIZE 1 \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}
done
done
