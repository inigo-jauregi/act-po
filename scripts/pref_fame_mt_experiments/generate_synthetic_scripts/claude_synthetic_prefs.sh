#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4
export AWS_PROFILE=<AWS_PROFILE>

EXPERIMENT_NAME="<EXPERIMENT_NAME>"
SRC_LANG="da"
TGT_LANG="es"

echo "Generate preference data for ${SRC_LANG} to ${TGT_LANG}"
python3 -m scripts.pref_fame_mt_experiments.generate_synthetic_data \
    --experiment-name ${EXPERIMENT_NAME} \
    --src ${SRC_LANG} \
    --tgt ${TGT_LANG} \
    --fixed 'rej'
