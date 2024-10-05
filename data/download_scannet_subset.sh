#!/bin/bash

OUTPUT_DIR="data/scannet_scenes/"  

FILE_TYPES=(
    "_vh_clean.aggregation.json"
    "_vh_clean.ply"
    "_vh_clean.segs.json"
)

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <start_scene> <stop_scene>"
  echo "Example: $0 0 49"
  exit 1
fi

START_SCENE=$1
STOP_SCENE=$2

for i in $(seq -w $START_SCENE $STOP_SCENE)
do
    SCENE_ID="scene$(printf "%04d" $i)_00"
    echo "Downloading files for ${SCENE_ID}..."

    for TYPE in "${FILE_TYPES[@]}"
    do
        echo "  Downloading ${TYPE}..."
        echo | python3 data/download_scannet.py --type $TYPE --id $SCENE_ID --o $OUTPUT_DIR
    done
done
