#!/bin/bash

RAG_ENCODER_TYPE="${RAG_ENCODER_TYPE:-clip_encoder}"
RAG_ENCODER_PATH="${RAG_ENCODER_PATH:-}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/data/3/MedPLIB/dataset/images-and-masks-root}"
QUERY_JSON="${QUERY_JSON:-/data/3/MedPLIB/dataset/MedPLIB_ICL_test.json}"
RAG_OUTPUT_JSON="${RAG_OUTPUT_JSON:-/data/3/MedPLIB/dataset/MedPLIB_ICL_RAG_test.json}"
RAG_INDEX_DIR="${RAG_INDEX_DIR:-/data/3/MedPLIB/dataset/rag_index}"
RAG_TOP_K="${RAG_TOP_K:-3}"
RAG_BATCH_SIZE="${RAG_BATCH_SIZE:-16}"
RAG_PRECISION="${RAG_PRECISION:-bf16}"

python model/rag/image_rag.py augment \
  --rag_encoder_type "$RAG_ENCODER_TYPE" \
  --rag_encoder_path "$RAG_ENCODER_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --query_json "$QUERY_JSON" \
  --output_json "$RAG_OUTPUT_JSON" \
  --index_dir "$RAG_INDEX_DIR" \
  --top_k "$RAG_TOP_K" \
  --batch_size "$RAG_BATCH_SIZE" \
  --precision "$RAG_PRECISION"

VAL_DATA_PATH="$RAG_OUTPUT_JSON" sh scripts/infer_medplib_icl.sh
