
time=$(date +%Y-%m-%d-%H-%M-%S)
export NCCL_DEBUG=INFO

exp_name="medplib-7b-stage3"
exp_dir="runs/$exp_name"
mkdir -p "$exp_dir"

TRANSFORMERS_OFFLINE=1 deepspeed --include=localhost:0,1,2,3 --master_port=64999 train_ds_medplib.py \
  --version="/root/paddlejob/workspace/env_run/output/huangxiaoshuang/MedPLIB/runs/medplib-7b-stage2/hf" \
  --vision_tower='/root/paddlejob/workspace/env_run/output/huangxiaoshuang/huggingface_models/clip-vit-large-patch14-336' \
  --data_path='/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/MeCoVQA/MeCoVQA_train.json' \
  --val_data_path='/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/MeCoVQA/MeCoVQA_Grounding_test_rand500.json' \
  --image_folder='/tmp/v2_mnt/HCG/huangxiaoshuang/SAMed2D_v1' \
  --vision_pretrained="/root/paddlejob/workspace/env_run/output/huangxiaoshuang/huggingface_models/sam-med2d_b.pth" \
  --exp_name=$exp_name \
  --epochs=3 \
  --batch_size=4 \
  --workers=2 \
  --image_aspect_ratio='pad' \
  --is_multimodal=True \
  --model_max_length 1024 \
  --grad_accumulation_steps 8 \
  --out_dim 256 \
  --ce_loss_weight 1.0 \
  --dice_loss_weight 0.5 \
  --bce_loss_weight 2.0 \
  --focal_loss_weight 0 \
  --iou_loss_weight 0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_target_modules "gate_proj,up_proj,down_proj,q_proj,v_proj" \
  --sft_modules "wg,lm_head,embed_tokens,mask_decoder,text_hidden_fcs,region_fea_adapter" \
  --lr 0.0003 \
  --save_steps 500 \
  --sam_img_size 256 \
  --train_mask_decoder \
  --moe_enable ture \
  --moe_mode dense \
  --num_experts 2 \
  --capacity_factor 1.5 \
  --region_fea_adapter \
  --top_k_experts 1 \
  --router_aux_loss_coef 0.0 \
  --expert_pretrained_path "/root/paddlejob/workspace/env_run/output/huangxiaoshuang/MedPLIB/runs/medplib-7b-stage3/hf,/root/paddlejob/workspace/env_run/output/huangxiaoshuang/MedPLIB/runs/medplib-7b-stage2/hf" \
  --no_eval \
  2>&1|tee -a runs/$exp_name/$time.log
