#!/bin/bash

# custom config
DATA=../datasets # data root
TRAINER=MaskTuning

#  =================
DATASET=$1  # oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
GDR_LAMBDA=$2 #$ # use cosine lr scheduler
#  =================
CFG=vit_b16
if [[ ${DATASET} == imagenet ]]
then
	CFG_FILE=imagenet_${CFG}.yaml
else
	CFG_FILE=${CFG}.yaml
fi

SHOTS=16  # number of shots (1, 2, 4, 8, 16)
for SEED in 1 2 3
do
    DIR=../output_ramt/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
   	python train.py \
    	--root ${DATA} \
    	--seed ${SEED} \
    	--trainer ${TRAINER} \
    	--dataset-config-file configs/datasets/${DATASET}.yaml \
    	--config-file configs/trainers/MASK/${CFG_FILE} \
    	--output-dir ${DIR} \
		TEST.FINAL_MODEL "last_step" \
    	DATASET.NUM_SHOTS ${SHOTS} \
		MASK.GDR_LAMBDA ${GDR_LAMBDA}
    fi
done