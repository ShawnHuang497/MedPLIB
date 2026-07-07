#!/bin/bash

time=$(date +%Y-%m-%d-%H-%M-%S)
NCCL_DEBUG=WARN

# -------- Runtime config --------
GPU_IDS="${GPU_IDS:-0}"
MASTER_PORT="${MASTER_PORT:-65000}"

# overlay: example image with mask overlay as one image.
# separate: example image and example mask as two independent images.
ICL_MASK_MODE="${ICL_MASK_MODE:-overlay}"
ICL_EXTRA_ARGS="--mm_token_compress --mm_compressed_token_count 256"

if [ "$ICL_MASK_MODE" = "separate" ]; then
  ICL_EXTRA_ARGS="$ICL_EXTRA_ARGS --icl_mask_encoder --mask_encoder_token_count 64"
fi

# -------- Path config --------
CKPT_PATH="${CKPT_PATH:-/data/3/MedPLIB/checkpoint/medplib-icl-checkpoint}"
VISION_TOWER="${VISION_TOWER:-/data/3/MedPLIB/checkpoint/clip-vit-large-patch14-336}"
VISION_PRETRAINED="${VISION_PRETRAINED:-/data/3/MedPLIB/checkpoint/sam-med2d_b.pth}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/data/3/MedPLIB/dataset/images-and-masks-root}"
VAL_DATA_PATH="${VAL_DATA_PATH:-/data/3/MedPLIB/dataset/MedPLIB_ICL_test.json}"

EXP_NAME="${EXP_NAME:-medplib-icl-infer}"
LOG_DIR="runs/$EXP_NAME"
mkdir -p "$LOG_DIR"

# Use 4096 for overlay by default; separate mode may need 8192 for 3-shot.
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"

deepspeed --include=localhost:$GPU_IDS --master_port=$MASTER_PORT model/eval/vqa_infer.py \
  --icl_enable \
  --icl_mask_mode="$ICL_MASK_MODE" \
  $ICL_EXTRA_ARGS \
  --version="$CKPT_PATH" \
  --vision_tower="$VISION_TOWER" \
  --val_data_path="$VAL_DATA_PATH" \
  --image_folder="$IMAGE_FOLDER" \
  --vision_pretrained="$VISION_PRETRAINED" \
  --image_aspect_ratio="pad" \
  --is_multimodal \
  --model_max_length "$MODEL_MAX_LENGTH" \
  --sam_img_size 256 \
  --eval_seg \
  --moe_enable \
  --vis_mask \
  2>&1 | tee -a "$LOG_DIR/$time.log"
