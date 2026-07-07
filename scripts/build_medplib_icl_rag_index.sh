#!/bin/bash

RAG_ENCODER_TYPE="${RAG_ENCODER_TYPE:-clip_encoder}"
RAG_ENCODER_PATH="${RAG_ENCODER_PATH:-}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/data/3/MedPLIB/dataset/images-and-masks-root}"
CANDIDATE_JSON="${CANDIDATE_JSON:-/data/3/MedPLIB/dataset/MedPLIB_ICL_train.json}"
RAG_INDEX_DIR="${RAG_INDEX_DIR:-/data/3/MedPLIB/dataset/rag_index}"
RAG_BATCH_SIZE="${RAG_BATCH_SIZE:-16}"
RAG_PRECISION="${RAG_PRECISION:-bf16}"

python model/rag/image_rag.py build \
  --rag_encoder_type "$RAG_ENCODER_TYPE" \
  --rag_encoder_path "$RAG_ENCODER_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --candidate_json "$CANDIDATE_JSON" \
  --index_dir "$RAG_INDEX_DIR" \
  --batch_size "$RAG_BATCH_SIZE" \
  --precision "$RAG_PRECISION"
