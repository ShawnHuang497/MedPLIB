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
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="/root/huggingface_models/llava-v1.5-7b"
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
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision_tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)


    # 添加数据参数
    parser.add_argument('--image_folder', type=str, default='/tmp/v2_mnt/HCG/huangxiaoshuang/SAMed2D_v1', help='Path to the folder containing images.')
    parser.add_argument('--image_aspect_ratio', type=str, default='pad', help='How to handle image aspect ratio.')
    parser.add_argument('--is_multimodal', type=bool, default=True, help='Whether to use multimodal data.')
    parser.add_argument('--data_path', type=str, default='/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/seg/0301_CombinedAll_7.5w_train1_chatlm_img_coord2_rand5k_seg_cat_origin.json', help='Path to the JSON file containing the data.')
    parser.add_argument('--val_data_path', type=str, default='/root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/seg/0301_CombinedAll_7.5w_train1_chatlm_img_coord2_rand5k_seg.json', help='Path to the JSON file containing the data.')

    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    # parser.add_argument("--steps_per_epoch", default=500, type=int)
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
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--sft_modules", default="lm_head,embed_tokens,mask_decoder,text_hidden_fcs", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
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

    # ---------------rp sampler-----------
    parser.add_argument("--region_fea_adapter", action="store_true", default=False)
    parser.add_argument("--region_geo_sampler", action="store_true", default=False)
    parser.add_argument("--max_sample_point", default=512, type=int)
    parser.add_argument("--sampler_pooler_mode", default='max', type=str)

    # ------------------moe params-----------------
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

    # ----------------trained params -----------------

    parser.add_argument('--weight', type=str, default=None, help='path of trained weight')
    parser.add_argument('--save_path', type=str, default=None, help='path of trained weight')

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
    # global local_rank
    args = parse_args(args)
    # local_rank = args.local_rank
    # rank0_print("local rank:", local_rank)
    args.num_experts = [args.num_experts]
    set_seed(42)
    if args.local_rank == 0:
        test_randomness()

    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)

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
    args.seg_token_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
    # model_args = {
    #     "train_mask_decoder": args.train_mask_decoder,
    #     "out_dim": args.out_dim,
    #     "ce_loss_weight": args.ce_loss_weight,
    #     "dice_loss_weight": args.dice_loss_weight,
    #     "bce_loss_weight": args.bce_loss_weight,
    #     "iou_loss_weight": args.bce_loss_weight,
    #     "focal_loss_weight": args.bce_loss_weight,
    #     "seg_token_idx": args.seg_token_idx,
    #     "vision_pretrained": args.vision_pretrained,
    #     "vision_tower": args.vision_tower,
    #     "use_mm_start_end": args.use_mm_start_end,
    #     "region_fea_adapter": args.region_fea_adapter,
    #     "region_geo_sampler": args.region_geo_sampler,
    #     "max_sample_point": args.max_sample_point,
    #     "sampler_pooler_mode": args.sampler_pooler_mode,
    # }
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
        # load sam  and text_hidden_fcs weights
        model = LISAForCausalLM.from_pretrained(
            args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, ignore_mismatched_sizes=True,**model_args
        )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # load tower and projector weights
    # model.get_model().initialize_vision_modules(model.get_model().config)
    # if not args.eval_only:
        # load sam and text_hidden_fcs weights
    if args.moe_enable:
        model.get_model().initialize_bird_modules(model.get_model().config)
    else:
        model.get_model().initialize_lisa_modules(model.get_model().config)


    vision_tower = model.get_model().get_vision_tower()
    # vision_tower.to(dtype=torch_dtype, device=args.local_rank)

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
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                    # and any([x in name and 'moe' in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
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



    model.resize_token_embeddings(len(tokenizer))

    state_dict = torch.load(args.weight, map_location="cpu")
    print('args.weight', args.weight)
    for k,v in state_dict.items():
        print(k, v.shape)

    # remove visual_model if required
    # state_dict = {k: v for k, v in state_dict.items() if "visual_model" not in k}


    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print('unexpected_keys',unexpected_keys)
    print('missing_keys',missing_keys)

    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        # if "vision_tower" not in k:
        state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
