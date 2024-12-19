import argparse
import os
import shutil
import sys
import time
from functools import partial
import types
import math
import random

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LISA import LISAForCausalLM
from model.MedPLIB import MedPLIBForCausalLM
from model.medplib import conversation as conversation_lib
# from utils.dataset import HybridDataset, ValDataset, collate_fn
from datasets import LazySupervisedDataset, DataCollatorForSupervisedDataset
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU, ADD_OTHERS_TOKENS)

local_rank = None

def parse_args(args):
    parser = argparse.ArgumentParser(description="MedPLIB Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    # ----------------- Model setting
    parser.add_argument(
        "--version", default="/path/to/llava-v1.5-7b"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--pretrain_mm_mlp_adapter", default=None, type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--sam_img_size", default=256, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--vision_tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--sft_modules", default="lm_head,embed_tokens,mask_decoder,text_hidden_fcs", type=str, help='set the supervised fintuning modules')

    # ------------------ Lora setting -------------------
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)

    # ---------------- Dataset setting -----------------
    parser.add_argument('--image_folder', type=str, default='/path/to/SAMed2D_v1', help='Path to the folder containing images.')
    parser.add_argument('--image_aspect_ratio', type=str, default='pad', help='How to handle image aspect ratio.')
    parser.add_argument('--is_multimodal', type=bool, default=True, help='Whether to use multimodal data.')
    parser.add_argument('--data_path', type=str, default='/path/to/xxx.json', help='Path to the JSON file containing the data.')
    parser.add_argument('--val_data_path', type=str, default='/path/to/xxx.json', help='Path to the JSON file containing the data.')

    # ---------------- Training setting -----------------
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--iou_loss_weight", default=2.0, type=float)
    parser.add_argument("--focal_loss_weight", default=2.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--save_steps", default=10, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    # ---------------region sampler setting-----------
    parser.add_argument("--region_fea_adapter", action="store_true", default=False)
    parser.add_argument("--region_geo_sampler", action="store_true", default=False)
    parser.add_argument("--max_sample_point", default=512, type=int)
    parser.add_argument("--sampler_pooler_mode", default='max', type=str)

    # ------------------moe setting-----------------
    parser.add_argument('--moe_enable', type=bool, default=False, help='Use residual connections')
    parser.add_argument('--moe_mode', type=str, default='second_half',
                    choices=['first_half', 'second_half', 'sparse', 'dense'],
                    help='The backend to be used for half precision.')
    parser.add_argument('--num_experts', type=int, default=3, help='Number of experts')
    parser.add_argument('--top_k_experts', type=int, default=2, help='Top K experts to use')
    parser.add_argument('--capacity_factor', type=float, default=1, help='Capacity factor')
    parser.add_argument('--use_residual', type=bool, default=False, help='Use residual connections')
    parser.add_argument('--router_aux_loss_coef', type=float, default=0.01, help='Router auxiliary loss coefficient')
    parser.add_argument('--eval_capacity_factor', type=float, default=2, help='Evaluation capacity factor')
    parser.add_argument('--moe_layers_idx', type=str, default=None, help='Indices of layers to apply MOE')
    parser.add_argument('--min_capacity', type=int, default=0, help='Minimum capacity')
    parser.add_argument('--ep_size', type=int, default=1, help='EP size')
    parser.add_argument('--expert_pretrained_path', type=str, default=None, help='path of different MOE')
    parser.add_argument('--finetune_moe', type=bool, default=False, help='finetune on downstream task')
    return parser.parse_args(args)
        
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    random.seed(seed)            
    np.random.seed(seed)         
    torch.manual_seed(seed)      
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)  

def test_randomness():
    # Python random
    print("Python random:")
    print([random.randint(1, 100) for _ in range(5)])

    # NumPy random
    print("\nNumPy random:")
    print(np.random.rand(5))

    # PyTorch random
    print("\nPyTorch random:")
    print(torch.rand(5))

    # PyTorch layer weights
    linear = torch.nn.Linear(10, 5)
    print("PyTorch Linear weights:")
    print(linear.weight)


def main(args):
    global local_rank
    args = parse_args(args)
    local_rank = args.local_rank
    rank0_print("local rank:", local_rank)
    args.num_experts = [args.num_experts]
    set_seed(42)
    if args.local_rank == 0:
        test_randomness()

    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0 and not args.eval_only:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    for i in range(1, 257):
        ADD_OTHERS_TOKENS.append("<gen_" + str(i) + ">")
    for token_name in ADD_OTHERS_TOKENS:
        tokenizer.add_tokens(token_name, special_tokens=True)
    args.seg_token_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = vars(args)
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    if args.moe_enable:
        model = MedPLIBForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, ignore_mismatched_sizes=True,**model_args
    )
    else:
        # load LLM and tower weights
        model = LISAForCausalLM.from_pretrained(
            args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, ignore_mismatched_sizes=True,**model_args
        )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # load tower and projector weights
    model.get_model().initialize_vision_modules(model.get_model().config)
    if not args.eval_only:
        # load sam and text_hidden_fcs weights
        if args.moe_enable:
            model.get_model().initialize_bird_modules(model.get_model().config)
        else:
            model.get_model().initialize_lisa_modules(model.get_model().config)


    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                # "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        if args.local_rank == 0:
            print('lora_target_modules', lora_target_modules)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        for n, p in model.named_parameters():
            p.requires_grad = False
        
    if args.moe_enable:
        # load moe weights
        model.initialize_moe_modules(args)

    model.resize_token_embeddings(len(tokenizer))


    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    if args.sft_modules != "":
        sft_modules = args.sft_modules.split(",")
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in sft_modules
                ]
            ):
                # print("n: ", n, "p.shape: ", p.shape)
                p.requires_grad = True

    # if args.local_rank == 0:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(f"Parameter Name: {name}, Shape: {param.shape}, grad: {param.requires_grad}")

    if args.local_rank == 0:
        for name, param in model.named_parameters():
            print(f"Parameter Name: {name}, Shape: {param.shape}, grad: {param.requires_grad}")
            # print(param)
    # --------------show trainable params---------------
    def count_parameters(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params
    # count params
    trainable_params, total_params = count_parameters(model)
    trainable_params_percentage = trainable_params / total_params * 100
    if args.local_rank == 0:
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters Percentage: {trainable_params_percentage:.7f}%")
    # -------------------------------------------------


    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    data_args = {"image_folder": args.image_folder,
                "image_aspect_ratio": args.image_aspect_ratio,
                "is_multimodal": args.is_multimodal,
                "mm_use_im_start_end": args.use_mm_start_end,
                "data_path": args.data_path,
                "image_processor": vision_tower.image_processor
                }
    data_args = types.SimpleNamespace(**data_args)
    train_dataset = LazySupervisedDataset(args.data_path, tokenizer, data_args, args.sam_img_size)

    args.steps_per_epoch = math.ceil(math.ceil(len(train_dataset) / (args.batch_size * torch.cuda.device_count())) / args.grad_accumulation_steps)

    if args.no_eval == False:

        val_dataset = LazySupervisedDataset(args.val_data_path, tokenizer, data_args, args.sam_img_size)
        
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples. steps in one epoch: {args.steps_per_epoch}"
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples. steps in one epoch: {args.steps_per_epoch}")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "gradient_clipping": 1.0,
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": int(args.steps_per_epoch * 0.01),
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

    if args.moe_enable and 'up_proj' in args.lora_target_modules:
        from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
        def create_moe_param_groups(model):
            from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

            parameters = {'params': [p for p in model.parameters()], 'name': 'parameters'}

            return split_params_into_different_moe_groups_for_optimizer(parameters)
        optimizer_grouped_parameters = create_moe_param_groups(model)
        print('use moe and grad it!!!!')
    else:
        # 如果不使用 MOE，直接使用所有参数
        optimizer_grouped_parameters = model.parameters()

    
    

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=optimizer_grouped_parameters,
        training_data=train_dataset,
        # training_data=None,
        collate_fn=partial(
            DataCollatorForSupervisedDataset,
        ),
        config=ds_config,
    )
    print(f"deepspeed.initialize done!!!!!!!!!!")


    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        print('The resume model global_steps is ', model_engine.global_steps)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                DataCollatorForSupervisedDataset,
                inference=True,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        giou, ciou = validate(val_loader, model_engine, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )


        if args.no_eval == False:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        save_dir = os.path.join(args.log_dir, "last_ckpt_model")
        if args.local_rank == 0 and os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        torch.distributed.barrier()
        model_engine.save_checkpoint(save_dir)

def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.2f")
    data_time = AverageMeter("Data", ":6.2f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    unscale_mask_bce_losses = AverageMeter("unscale_mask_bce_loss", ":.4f")
    unscale_mask_dice_losses = AverageMeter("unscale_mask_dice_loss", ":.4f")
    unscale_mask_losses = AverageMeter("unscale_mask_loss", ":.4f")
    unscale_mask_iou_losses = AverageMeter("unscale_mask_iou_loss", ":.4f")
    unscale_mask_focal_losses = AverageMeter("unscale_mask_focal_loss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            model.global_steps,
            batch_time,
            data_time,
            losses,
            ce_losses,
            unscale_mask_losses,
            unscale_mask_bce_losses,
            unscale_mask_dice_losses,
            unscale_mask_iou_losses,
            unscale_mask_focal_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()

    if model.global_steps % args.steps_per_epoch > 0:
        print('skipping first {} steps, global step is {}'.format(model.global_steps % args.steps_per_epoch, model.global_steps))
        
        for local_step in tqdm.tqdm(range(0, model.global_steps % args.steps_per_epoch)):
            for i in range(args.grad_accumulation_steps):
                try:
                    input_dict = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    input_dict = next(train_iter)

    for local_step in range(model.global_steps % args.steps_per_epoch, args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            unscale_mask_bce_loss = output_dict["unscale_mask_bce_loss"]
            unscale_mask_dice_loss = output_dict["unscale_mask_dice_loss"]
            unscale_mask_loss = output_dict["unscale_mask_loss"]
            unscale_mask_iou_loss = output_dict["unscale_mask_iou_loss"]
            unscale_mask_focal_loss = output_dict["unscale_mask_focal_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            if input_dict['seg_flag']:
                mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
                mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
                mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
                unscale_mask_bce_losses.update(unscale_mask_bce_loss.item(), input_dict["images"].size(0))
                unscale_mask_dice_losses.update(unscale_mask_dice_loss.item(), input_dict["images"].size(0))
                unscale_mask_losses.update(unscale_mask_loss.item(), input_dict["images"].size(0))
                unscale_mask_iou_losses.update(unscale_mask_iou_loss.item(), input_dict["images"].size(0))
                unscale_mask_focal_losses.update(unscale_mask_focal_loss.item(), input_dict["images"].size(0))
            # compute gradient and do SGD step
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if local_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()
                unscale_mask_bce_losses.all_reduce()
                unscale_mask_dice_losses.all_reduce()
                unscale_mask_losses.all_reduce()
                unscale_mask_iou_losses.all_reduce()
                unscale_mask_focal_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(local_step + 1)
                writer.add_scalar("train/loss", losses.avg, model.global_steps)
                writer.add_scalar("train/ce_loss", ce_losses.avg, model.global_steps)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, model.global_steps
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, model.global_steps
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, model.global_steps)
                writer.add_scalar("train/unscale_mask_bce_losses", unscale_mask_bce_losses.avg, model.global_steps)
                writer.add_scalar("train/unscale_mask_dice_losses", unscale_mask_dice_losses.avg, model.global_steps)
                writer.add_scalar("train/unscale_mask_losses", unscale_mask_losses.avg, model.global_steps)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, model.global_steps
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, model.global_steps
                )
                # writer.add_scalar(
                #     "metrics/average_gradient", average_gradient, model.global_steps
                # )

                writer.add_scalar("train/unscale_mask_iou_losses", unscale_mask_iou_losses.avg, model.global_steps)
                writer.add_scalar("train/unscale_mask_focal_losses", unscale_mask_focal_losses.avg, model.global_steps)

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()
            unscale_mask_bce_losses.reset()
            unscale_mask_dice_losses.reset()
            unscale_mask_losses.reset()
            unscale_mask_iou_losses.reset()
            unscale_mask_focal_losses.reset()

        if model.global_steps != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], model.global_steps)


        if local_step != 0 and local_step % args.save_steps == 0:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0 and os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model.save_checkpoint(save_dir)

    return train_iter

def calculate_iou(prediction_mask, ground_truth_mask):
    # to boolen
    prediction_mask = prediction_mask.bool()
    ground_truth_mask = ground_truth_mask.bool()

    # cal i and u
    intersection = torch.logical_and(prediction_mask, ground_truth_mask)
    union = torch.logical_or(prediction_mask, ground_truth_mask)

    intersection_pixels = torch.sum(intersection)
    union_pixels = torch.sum(union)

    if union_pixels == 0:
        iou = 0
    else:
        iou = intersection_pixels.float() / union_pixels.float()

    return iou.item()

def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    iou_meter = AverageMeter("IoU", ":6.3f", Summary.SUM)
    dice_meter = AverageMeter("Dice", ":6.3f", Summary.SUM)

    model_engine.eval()

    pbar = tqdm.tqdm(val_loader)
    for input_dict in pbar:
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int().unsqueeze(0)
        output_list = (torch.sigmoid(pred_masks[0]) > 0.1).int()
        # assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            mask_i = mask_i.unsqueeze(0)
            output_i = output_i.unsqueeze(0)
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
            iou = calculate_iou(output_i, mask_i)
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

        iou_meter.update(iou)
        dice_meter.update(2*iou/(1+iou))

        new_description = f'iou={acc_iou}.'
        pbar.set_description(new_description)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    iou_meter.all_reduce()
    dice_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    miou = iou_meter.avg
    mDice = dice_meter.avg

    if args.local_rank == 0 and not args.eval_only:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        writer.add_scalar("val/dice", mDice, epoch)
    print("giou: {:.6f}, ciou: {:.6f}".format(giou, ciou))
    print("miou: {:.6f}, mDice: {:.6f}".format(miou, mDice))
    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])
