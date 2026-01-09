#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4

FIXED_SEED=42
BATCH_SIZE=4
MAX_SEQ_LEN=512
VAL_CHECK_INTERVAL=1000
MAX_EPOCHS=30
LEARNING_RATE=5e-7
MODEL_NAME="./pretrained_lms/utter-project-EuroLLM-9B-Instruct"
PROMPT_UNCTRL="Here is a sentence {<INPUT_SRC>}; Here is its <TGT_LANG> translation {<OUTPUT_TGT>};"
PROMPT_CTRL="Here is a sentence {<INPUT_SRC>}; Here is its <TGT_LANG> translation written in <FORMALITY> style {<OUTPUT_TGT>};
The translated sentence conveys a <FORMALITY> style by using words such as <FORMALITY_TOKENS>."
PRETRAINED="<PRETRAINED_PATH>"  # ending in '_LoRA'
BETA=1
LAMBDA=0.25

SRC_LANG=("da")
TGT_LANG=("es")

for i in "${!SRC_LANG[@]}"; do

    TRAIN_PATH="./data/PREF_FAME_MT/${SRC_LANG[i]}-${TGT_LANG[i]}/train_contrastive.csv"
    VAL_PATH="./data/PREF_FAME_MT/${SRC_LANG[i]}-${TGT_LANG[i]}/val_contrastive.csv"
    TEST_PATH="./data/PREF_FAME_MT/${SRC_LANG[i]}-${TGT_LANG[i]}/test_contrastive.csv"

    EXPERIMENT_NAME="PAPER_pref_fame_mt_${SRC_LANG[i]}-${TGT_LANG[i]}"

    echo "EXPERIMENT: ${EXPERIMENT_NAME} | Qwen3 (IT+CPO)"
    python3 -m scripts.pref_fame_mt_experiments.train_model \
      --model-name ${MODEL_NAME} \
      --train-data ${TRAIN_PATH} \
      --val-data ${VAL_PATH} \
      --test-data ${TEST_PATH} \
      --experiment-name ${EXPERIMENT_NAME} \
      --src-lng ${SRC_LANG[i]} \
      --tgt-lng ${TGT_LANG[i]} \
      --prompt "${PROMPT_CTRL}" \
      --objective cpo \
      --dpo-beta ${BETA} \
      --cpo-lambda ${LAMBDA} \
      --val-check-interval ${VAL_CHECK_INTERVAL} \
      --max-length ${MAX_SEQ_LEN} \
      --max-epochs ${MAX_EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --padding-side "left" \
      --learning-rate ${LEARNING_RATE} \
      --fix-seed ${FIXED_SEED} \
      --strategy "ddp"  \
      --overwrite-arguments \
      --from-pretrained ${PRETRAINED} \
      --synthetic-pref-data ./data/PREF_FAME_MT/${SRC_LANG[i]}-${TGT_LANG[i]}/synthetic/preference_train_data/sonnet_4.5_ic_sampling_temp_ref_rej/synthetic_pref_data.csv  \
      --only-synthetic

done
