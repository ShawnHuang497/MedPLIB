
#!/bin/bash
CUDA_VISIBLE_DEVICES="0,1,2,3"
# gpu_list="0,0,0,1,1,1,2,2,2,3,3,3"
gpu_list="0,0,1,1,2,2,3,3"
# gpu_list="0,1,2,3"
echo GPU_LIST: $gpu_list
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo CHUNKS: $CHUNKS

CKPT="MedPLIB-7b-2e"

####################data config################

### For MeCoVQA
# ROOT_PATH="/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons"

DATA_TYPE="MeCoVQA"
SPLIT="MeCoVQA_Complex_VQA_test"
# SPLIT="MeCoVQA_Region_VQA_test"

### For public dataset
ROOT_PATH="/tmp/v2_mnt/HCG/huangxiaoshuang/med-vqa-dataset"

# DATA_TYPE="VQA-RAD"
# SPLIT="VQA_RAD_llavaformat_test_closed"
# SPLIT="VQA_RAD_llavaformat_test_open"
# SPLIT="train_llavaformat_oneturn_closed"
# SPLIT="train_llavaformat_oneturn_open"


# DATA_TYPE="Slake1.0"
# SPLIT="train_llavaformat_test_closed"
# SPLIT="train_llavaformat_test_open"

# DATA_TYPE="path-vqa"
# SPLIT="test_llavaformat_closed"
# SPLIT="test_llavaformat_open"

# DATA_TYPE="ImageClef-2019-VQA-Med"
# SPLIT="test_llavaformat_oneturn_closed"
# SPLIT="test_llavaformat_oneturn_open"

# DATA_TYPE="ImageClef-2021-VQA-Med"
# SPLIT="val_llavaformat_oneturn_closed"
# SPLIT="val_llavaformat_oneturn_open"

# DATA_TYPE="PMC-VQA-v2"
# SPLIT="PMC-VQA-v2-llavaformat_test_2_oneturn"

DATA_TYPE="OmniMedVQA"
# SPLIT="OmniMedVQA_OA_rand200"
# SPLIT="OmniMedVQA_OA_rand_all2000"
SPLIT="OmniMedVQA_OA"

### Control the answer type
answer_type='closed'
# answer_type='open'

PORT=64500
for IDX in $(seq 0 $((CHUNKS-1))); do
    PORT=$((PORT-1))
    deepspeed --include=localhost:${GPULIST[$IDX]} --master_port=$PORT model/eval/vqa_infer.py \
    --version="/public/home/s20213081508/huangxiaoshuang/huggingface/$CKPT" \
    --vision_tower='/public/home/s20213081508/huangxiaoshuang/huggingface/clip-vit-large-patch14-336' \
    --answer_type=$answer_type \
    --image_folder='/public/home/s20213081508/huangxiaoshuang/data/SA-2D-20M/GMAI___SA-Med2D-20M/raw/SAMed2Dv1' \
    --vision_pretrained="/public/home/s20213081508/huangxiaoshuang/huggingface/sam-med2d_b.pth" \
    --val_data_path $ROOT_PATH/$DATA_TYPE/$SPLIT.json \
    --answers-file /public/home/s20213081508/huangxiaoshuang/MedPLIB/runs/$CKPT/infer_res/$DATA_TYPE/$SPLIT/${CHUNKS}_${IDX}.jsonl \
    --eval_vqa \
    --region_fea_adapter \
    --moe_enable \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX &
done

wait

output_file=/public/home/s20213081508/huangxiaoshuang/MedPLIB/runs/$CKPT/infer_res/$DATA_TYPE/$SPLIT.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /public/home/s20213081508/huangxiaoshuang/MedPLIB/runs/$CKPT/infer_res/$DATA_TYPE/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

