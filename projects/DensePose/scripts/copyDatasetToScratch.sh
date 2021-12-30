#!/bin/bash

UNIQUEID=$(uuidgen)
UNIQUEID=${UNIQUEID:0:13}

mkdir /scratch0/$USER/
BASEDIR="/scratch0/$USER"

COPYDIR="${BASEDIR}/${UNIQUEID}"
mkdir $COPYDIR

rsync -ar --info=progress2 /SAN/medic/recons3d/datasets/AMASS/coco.zip $COPYDIR

cd $COPYDIR

unzip -q coco.zip
rm coco.zip
