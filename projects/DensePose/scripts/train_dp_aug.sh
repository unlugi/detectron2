#!/usr/bin/env bash

OUTPUT_PATH=${1} # DIRECTORY TO SAVE EXPERIMENT RESULTS
PROB_JOINTS=${2}
PROB_LIMBS_OCC=${3}
PROB_LIMBS_VIS=${4}
PROB_TORSO_OCC=${5}
PROB_TORSO_VIS=${6}

# optionally activate your conda env here:
#conda activate dtn

CUDA_VISIBLE_DEVICES=0 python3 train_net.py --config-file configs/sketch/densepose_rcnn_R_50_FPN_s1x_legacy.yaml \
                                                      MODEL.KEYPOINT_ON True \
                                                      SOLVER.IMS_PER_BATCH 4 \
                                                      SOLVER.BASE_LR 0.001 \
                                                      DATALOADER.NUM_WORKERS 2 \
                                                      INPUT.SVG.AUG_PROB_JOINTS ${PROB_JOINTS} \
                                                      INPUT.SVG.AUG_PROB_LIMBS_OCC ${PROB_LIMBS_OCC} \
                                                      INPUT.SVG.AUG_PROB_LIMBS_VIS ${PROB_LIMBS_VIS} \
                                                      INPUT.SVG.AUG_PROB_TORSO_OCC ${PROB_TORSO_OCC} \
                                                      INPUT.SVG.AUG_PROB_TORSO_VIS ${PROB_TORSO_VIS} \
                                                      OUTPUT_DIR ${OUTPUT_PATH} \
# Print hostname and date for reference again
hostname
date

# give time for a clean exit.
sleep 10

date
