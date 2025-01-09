

# ---------------- LISA vqa -----------------
TRANSFORMERS_OFFLINE=1 deepspeed --include=localhost:0 --master_port=64999 model/eval/vqa_infer.py \
    --version="/root/paddlejob/workspace/env_run/output/huangxiaoshuang/MedPLIB/runs/LISA_Med-7B/hf" \
    --vision_tower='/root/paddlejob/workspace/env_run/output/huangxiaoshuang/huggingface_models/clip-vit-large-patch14-336' \
    --answer_type='open' \
    --val_data_path='/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/MeCoVQA/MeCoVQA_Complex_VQA_test.json' \
    --image_folder='/tmp/v2_mnt/HCG/huangxiaoshuang/SAMed2D_v1' \
    --vision_pretrained="/tmp/v2_mnt/HCG/open_source_model/sam_vit_h_4b8939.pth" \
    --eval_vqa \
    --sam_img_size 1024 \


# ---------------- LISA seg -----------------
TRANSFORMERS_OFFLINE=1 deepspeed --include=localhost:0 --master_port=64997 model/eval/vqa_infer.py \
    --version="/root/paddlejob/workspace/env_run/output/huangxiaoshuang/MedPLIB/runs/LISA_Med-7B/hf" \
    --vision_tower='/root/paddlejob/workspace/env_run/output/huangxiaoshuang/huggingface_models/clip-vit-large-patch14-336' \
    --answer_type='open' \
    --val_data_path='/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/MeCoVQA/MeCoVQA_Grounding_test.json' \
    --image_folder='/tmp/v2_mnt/HCG/huangxiaoshuang/SAMed2D_v1' \
    --vision_pretrained="/tmp/v2_mnt/HCG/open_source_model/sam_vit_h_4b8939.pth" \
    --eval_seg \
    --sam_img_size 1024 \



# ---------------------- stage2 ----------------------
TRANSFORMERS_OFFLINE=1 deepspeed --include=localhost:0 --master_port=64998 model/eval/vqa_infer.py \
    --version="/tmp/v2_mnt/HCG/huangxiaoshuang/checkpoints/MedPLIB/medplib-7b-stage2/hf" \
    --vision_tower='/root/paddlejob/workspace/env_run/output/huangxiaoshuang/huggingface_models/clip-vit-large-patch14-336' \
    --answer_type='open' \
    --val_data_path='/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/MeCoVQA/MeCoVQA_Complex_VQA_test.json' \
    --image_folder='/tmp/v2_mnt/HCG/huangxiaoshuang/SAMed2D_v1' \
    --vision_pretrained="/tmp/v2_mnt/HCG/open_source_model/sam-med2d_b.pth" \
    --eval_vqa \
    --sam_img_size 256 \


# ---------------------- stage3 ----------------------
TRANSFORMERS_OFFLINE=1 deepspeed --include=localhost:0 --master_port=64998 model/eval/vqa_infer.py \
    --version="/tmp/v2_mnt/HCG/huangxiaoshuang/checkpoints/MedPLIB/medplib-7b-stage2/hf" \
    --vision_tower='/root/paddlejob/workspace/env_run/output/huangxiaoshuang/huggingface_models/clip-vit-large-patch14-336' \
    --answer_type='open' \
    --val_data_path='/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/MeCoVQA/MeCoVQA_Grounding_test.json' \
    --image_folder='/tmp/v2_mnt/HCG/huangxiaoshuang/SAMed2D_v1' \
    --vision_pretrained="/tmp/v2_mnt/HCG/open_source_model/sam-med2d_b.pth" \
    --eval_seg \



# ---------------- stage4 vqa-----------------
TRANSFORMERS_OFFLINE=1 deepspeed --include=localhost:0 --master_port=64998 model/eval/vqa_infer.py \
    --version="/root/paddlejob/workspace/env_run/output/huangxiaoshuang/MedPLIB/runs/medplib-7b-stage4/hf" \
    --vision_tower='/root/paddlejob/workspace/env_run/output/huangxiaoshuang/huggingface_models/clip-vit-large-patch14-336' \
    --answer_type='open' \
    --val_data_path='/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/MeCoVQA/MeCoVQA_Complex_VQA_test.json' \
    --image_folder='/tmp/v2_mnt/HCG/huangxiaoshuang/SAMed2D_v1' \
    --vision_pretrained="/root/paddlejob/workspace/env_run/output/huangxiaoshuang/huggingface_models/sam-med2d_b.pth" \
    --eval_vqa \
    --moe_enable \
    --region_fea_adapter \
    --return_gating_logit \




# ---------------------- stage4 seg ----------------------
TRANSFORMERS_OFFLINE=1 deepspeed --include=localhost:0 --master_port=64995 model/eval/vqa_infer.py \
    --version="/root/paddlejob/workspace/env_run/output/huangxiaoshuang/MedPLIB/runs/medplib-7b-stage4/hf" \
    --vision_tower='/root/paddlejob/workspace/env_run/output/huangxiaoshuang/huggingface_models/clip-vit-large-patch14-336' \
    --answer_type='open' \
    --val_data_path='/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/MeCoVQA/MeCoVQA_Grounding_test.json' \
    --image_folder='/tmp/v2_mnt/HCG/huangxiaoshuang/SAMed2D_v1' \
    --vision_pretrained="/root/paddlejob/workspace/env_run/output/huangxiaoshuang/huggingface_models/sam-med2d_b.pth" \
    --eval_seg \
    --moe_enable \
    --region_fea_adapter \
    # --return_gating_logit \
    # --vis_mask \

TRANSFORMERS_OFFLINE=1 deepspeed --include=localhost:0 --master_port=64995 model/eval/vqa_infer.py \
    --version="/public/home/s20213081508/huangxiaoshuang/huggingface/MedPLIB-7b-2e" \
    --vision_tower='/public/home/s20213081508/huangxiaoshuang/huggingface/clip-vit-large-patch14-336' \
    --answer_type='open' \
    --val_data_path='/public/home/s20213081508/huangxiaoshuang/MedPLIB/MeCoVQA/MeCoVQA_Grounding_test.json' \
    --image_folder='/public/home/s20213081508/huangxiaoshuang/data/SA-2D-20M/GMAI___SA-Med2D-20M/raw/SAMed2Dv1' \
    --vision_pretrained="/public/home/s20213081508/huangxiaoshuang/huggingface/sam-med2d_b.pth" \
    --eval_seg \
    --moe_enable \
    --region_fea_adapter \
