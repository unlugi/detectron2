#!/usr/bin/env bash

# optionally activate your conda env here:
#conda activate dtn

#cd /home/gunlu/Projects/detectron2/projects/DensePose

OUTPUT_FOLDER=/mnt/data_s/gunlu/densepose/experiments

CUDA_VISIBLE_DEVICES=2 python3 train_net.py --config-file configs/sketch/densepose_rcnn_R_50_FPN_s1x_legacy.yaml \
                                            --eval-only \
                                            MODEL.WEIGHTS "$OUTPUT_FOLDER/output/model_final.pth" \
                                            OUTPUT_DIR "$OUTPUT_FOLDER/output" \
                                            MODEL.KEYPOINT_ON True \
# Give it a little time
sleep 10

CUDA_VISIBLE_DEVICES=2 python3 train_net.py --config-file configs/sketch/densepose_rcnn_R_50_FPN_s1x_legacy.yaml \
                                            --eval-only \
                                            MODEL.WEIGHTS "$OUTPUT_FOLDER/output2/model_final.pth" \
                                            OUTPUT_DIR "$OUTPUT_FOLDER/output2" \
                                            MODEL.KEYPOINT_ON True \

# Print hostname and date for reference again
hostname
date

# give time for a clean exit.
sleep 10

date
