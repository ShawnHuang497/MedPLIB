time=$(date +%Y-%m-%d-%H-%M-%S)
NCCL_DEBUG=WARN
exp_name="medplib-icl"
exp_dir="runs/$exp_name"
mkdir -p "$exp_dir"
ICL_MASK_MODE="${ICL_MASK_MODE:-overlay}"
ICL_EXTRA_ARGS="--mm_token_compress --mm_compressed_token_count 256"
SFT_MODULES="mask_decoder,text_hidden_fcs,mm_token_compressor"

if [ "$ICL_MASK_MODE" = "separate" ]; then
  ICL_EXTRA_ARGS="$ICL_EXTRA_ARGS --icl_mask_encoder --mask_encoder_token_count 64"
  SFT_MODULES="mask_decoder,text_hidden_fcs,mask_encoder,mm_token_compressor"
fi

deepspeed --include=localhost:0,1,2,3 --master_port=65000 train_ds_medplib.py \
  --icl_enable \
  --icl_mask_mode="$ICL_MASK_MODE" \
  $ICL_EXTRA_ARGS \
  --version="/data/3/MedPLIB/checkpoint/medplib" \
  --vision_tower="/data/3/MedPLIB/checkpoint/clip-vit-large-patch14-336" \
  --data_path="/data/3/MedPLIB/dataset/MedPLIB_ICL_train.json" \
  --val_data_path="/data/3/MedPLIB/dataset/MedPLIB_ICL_val.json" \
  --image_folder="/data/3/MedPLIB/dataset/images-and-masks-root" \
  --vision_pretrained="/data/3/MedPLIB/checkpoint/sam-med2d_b.pth" \
  --exp_name=$exp_name \
  --epochs=10 \
  --batch_size=4 \
  --workers=8 \
  --image_aspect_ratio="pad" \
  --is_multimodal=True \
  --model_max_length 4096 \
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
  --sft_modules "$SFT_MODULES" \
  --lr 0.0003 \
  --save_steps 300 \
  --sam_img_size 256 \
  --train_mask_decoder \
  --moe_enable True \
  2>&1 | tee -a runs/$exp_name/$time.log
