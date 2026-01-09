#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4

FIXED_SEED=42
BATCH_SIZE=4
MAX_SEQ_LEN=512
VAL_CHECK_INTERVAL=1000
MAX_EPOCHS=30
MODEL_NAME="./pretrained_lms/Qwen-Qwen3-8B"
PROMPT_UNCTRL="Here is a sentence {<INPUT_SRC>}; Here is its <TGT_LANG> translation {<OUTPUT_TGT>};"
PROMPT_CTRL="Here is a sentence {<INPUT_SRC>}; Here is its <TGT_LANG> translation written in <FORMALITY> style {<OUTPUT_TGT>};
The translated sentence conveys a <FORMALITY> style by using words such as <FORMALITY_TOKENS>."

SRC_LANG=("da")
TGT_LANG=("es")

for i in "${!SRC_LANG[@]}"; do

    TRAIN_PATH="./data/PREF_FAME_MT/${SRC_LANG[i]}-${TGT_LANG[i]}/train_contrastive.csv"
    VAL_PATH="./data/PREF_FAME_MT/${SRC_LANG[i]}-${TGT_LANG[i]}/val_contrastive.csv"
    TEST_PATH="./data/PREF_FAME_MT/${SRC_LANG[i]}-${TGT_LANG[i]}/test_contrastive.csv"

    EXPERIMENT_NAME="PAPER_pref_fame_mt_${SRC_LANG[i]}-${TGT_LANG[i]}"

    # zero-shot un-controlled experiments
    echo "EXPERIMENT: ${EXPERIMENT_NAME} | Qwen3 (ZS)"
    python3 -m scripts.pref_fame_mt_experiments.train_model \
        --model-name ${MODEL_NAME} \
        --train-data ${TRAIN_PATH} \
        --val-data ${VAL_PATH} \
        --test-data ${TEST_PATH} \
        --experiment-name ${EXPERIMENT_NAME} \
        --src-lng ${SRC_LANG[i]} \
        --tgt-lng ${TGT_LANG[i]} \
        --just-test \
        --padding-side "left" \
        --prompt "${PROMPT_CTRL}" \
        --max-length ${MAX_SEQ_LEN} \
        --batch-size ${BATCH_SIZE} \
        --fix-seed ${FIXED_SEED}

    echo "EXPERIMENT: ${EXPERIMENT_NAME} | Qwen3-LoRA (IT)"
    python3 -m scripts.pref_fame_mt_experiments.train_model \
        --model-name ${MODEL_NAME} \
        --train-data ${TRAIN_PATH} \
        --val-data ${VAL_PATH} \
        --test-data ${TEST_PATH} \
        --experiment-name ${EXPERIMENT_NAME} \
        --src-lng ${SRC_LANG[i]} \
        --tgt-lng ${TGT_LANG[i]} \
        --prompt "${PROMPT_CTRL}" \
        --max-length ${MAX_SEQ_LEN} \
        --padding-side "left" \
        --lora-finetuning \
        --log-model-mlflow \
        --max-epochs ${MAX_EPOCHS} \
        --fix-seed ${FIXED_SEED} \
        --val-check-interval ${VAL_CHECK_INTERVAL} \
        --devices "[0]" \
        --strategy "ddp"  # Needed for multi-GPU training and validation monitoring

done
