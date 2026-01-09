#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.4
export AWS_PROFILE=<AWS_PROFILE>

MODEL_SONNET_4_5="au.anthropic.claude-sonnet-4-5-20250929-v1:0"
# prompt_claude = "Here is a sentence {<INPUT_SRC>}; Please provide the <TGT_LANG> translation between {}: {<OUTPUT_TGT>};"
PROMPT="Here is a sentence {<INPUT_SRC>}; Please provide the <TGT_LANG> translation written in <FORMALITY> style between curly brackets: {<OUTPUT_TGT>};
        The translated sentence conveys a formal style by using words such as <FORMALITY_TOKENS>."
FIXED_SEED=42

SRC_LANG=("da")
TGT_LANG=("es")

for i in "${!SRC_LANG[@]}"; do
    for TEMP in 0.0 0.2 0.4 0.5 0.6 0.8 0.9 1.0; do
      for ITER in 1 2 3 4; do


        TRAIN_PATH="./data/PRE_FAME_MT/${SRC_LANG[i]}-${TGT_LANG[i]}/train_contrastive.csv"
        VAL_PATH="./data/PRE_FAME_MT/${SRC_LANG[i]}-${TGT_LANG[i]}/val_contrastive.csv"
        TEST_PATH="./data/PRE_FAME_MT/${SRC_LANG[i]}-${TGT_LANG[i]}/train_contrastive.csv"

        # zero-shot un-controlled experiments
        if [[ "${SRC_LANG[i]}" != "${TGT_LANG[i]}" ]]; then
          EXPERIMENT_NAME="synthetic_good_pref_fame_mt_sonnet_4.5_ic_${SRC_LANG[i]}-${TGT_LANG[i]}"
          echo "SAMPLE GENERATION for ${SRC_LANG} to ${TGT_LANG} - Iter ${ITER}"
          python3 -m scripts.pref_fame_mt_experiments.train_model \
              --model-name ${MODEL_SONNET_4_5} \
              --experiment-name ${EXPERIMENT_NAME} \
              --train-data ${TRAIN_PATH} \
              --num-test-samples 2000 \
              --val-data ${VAL_PATH} \
              --test-data ${TEST_PATH} \
              --src-lng ${SRC_LANG[i]} \
              --tgt-lng ${TGT_LANG[i]} \
              --just-test \
              --prompt "${PROMPT}" \
              --s3-bucket "ctrlpost-bedrock-inference-bucket" \
              --iam-role "arn:aws:iam::209378968454:role/ctrlpost-bedrock-inference-role" \
              --max-length 2048 \
              --in-context-learning \
              --in-context-num-samples 8 \
              --temperature ${TEMP} \
              --fix-seed ${FIXED_SEED} \
              --devices '[0]'
        fi
done
done
done