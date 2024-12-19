
time=$(date +%Y-%m-%d-%H-%M-%S)
NCCL_DEBUG=WARN
exp_name="medplib-7b-stage3"
exp_dir="runs/$exp_name"
mkdir -p "$exp_dir"

deepspeed --include=localhost:0,1,2,3 --master_port=65000 train_ds_medplib.py \
  --version="/root/paddlejob/workspace/env_run/output/huangxiaoshuang/MedPLIB/runs/medplib-7b-stage2/hf" \
  --vision_tower='/root/paddlejob/workspace/env_run/output/huangxiaoshuang/huggingface_models/clip-vit-large-patch14-336' \
  --data_path='/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/MeCoVQA/MeCoVQA_Grounding_train.json' \
  --val_data_path='/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/MeCoVQA/MeCoVQA_Grounding_test_rand500.json' \
  --image_folder='/tmp/v2_mnt/HCG/huangxiaoshuang/SAMed2D_v1' \
  --vision_pretrained="/root/paddlejob/workspace/env_run/output/huangxiaoshuang/huggingface_models/sam-med2d_b.pth" \
  --exp_name=$exp_name \
  --epochs=10 \
  --batch_size=32 \
  --workers=16 \
  --image_aspect_ratio='pad' \
  --is_multimodal=True \
  --model_max_length 2048 \
  --grad_accumulation_steps 1 \
  --out_dim 256 \
  --ce_loss_weight 1.0 \
  --dice_loss_weight 5.0 \
  --bce_loss_weight 1.0 \
  --iou_loss_weight 0 \
  --focal_loss_weight 1.0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_target_modules "gate_proj,up_proj,down_proj" \
  --sft_modules "mask_decoder,text_hidden_fcs" \
  --lr 0.0003 \
  --save_steps 300 \
  --sam_img_size 256 \
  --train_mask_decoder \
  --eval_only \
  2>&1|tee -a runs/$exp_name/$time.log
