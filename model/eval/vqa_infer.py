import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import shutil
import time
from functools import partial
import types
import math
import random
import shortuuid
import json

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import cv2
from torch.utils.data import Dataset, Subset

from model.LISA import LISAForCausalLM
from model.MedPLIB import MedPLIBForCausalLM
from model.medplib import conversation as conversation_lib
from datasets import LazySupervisedDataset, DataCollatorForSupervisedDataset
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU, ADD_OTHERS_TOKENS)



def parse_args(args):
    parser = argparse.ArgumentParser(description="Bird Model Testing")
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
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument(
        "--vision_tower", default="openai/clip-vit-large-patch14", type=str
    )


    # data setting
    parser.add_argument('--image_folder', type=str, default='/tmp/v2_mnt/HCG/huangxiaoshuang/SAMed2D_v1', help='Path to the folder containing images.')
    parser.add_argument('--image_aspect_ratio', type=str, default='pad', help='How to handle image aspect ratio.')
    parser.add_argument('--is_multimodal', action='store_true', default=True, help='Whether to use multimodal data.')
    parser.add_argument('--val_data_path', type=str, default='/tmp/v2_mnt/HCG/huangxiaoshuang/med-vqa-dataset/ImageClef-2019-VQA-Med/test_llavaformat_oneturn_open.json', help='Path to the JSON file containing the data.')
    parser.add_argument('--answer_type', type=str, default='closed', help='answer_type.')


    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--iou_loss_weight", default=2.0, type=float)
    parser.add_argument("--focal_loss_weight", default=2.0, type=float)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)


    # -----------------infer-------------------
    parser.add_argument('--eval_seg', action='store_true', default=False, help='')
    parser.add_argument('--eval_vqa', action='store_true', default=False, help='')

    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    parser.add_argument('--cpu_only', action='store_true', default=False, help='')
    parser.add_argument('--vis_mask', action='store_true', default=False, help='')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default='')

    # ---------------rp sampler-----------
    parser.add_argument("--region_fea_adapter", action="store_true", default=False)
    parser.add_argument("--region_geo_sampler", action="store_true", default=False)
    parser.add_argument("--max_sample_point", default=512, type=int)
    parser.add_argument("--sampler_pooler_mode", default='max', type=str)

    # ------------------moe params-----------------
    parser.add_argument('--moe_enable', action='store_true', default=False, help='Use residual connections')
    parser.add_argument('--moe_mode', type=str, default='second_half',
                    choices=['first_half', 'second_half', 'sparse', 'dense'],
                    help='The backend to be used for half precision.')
    parser.add_argument('--num_experts', type=int, default=3, help='Number of experts')
    parser.add_argument('--top_k_experts', type=int, default=2, help='Top K experts to use')
    parser.add_argument('--capacity_factor', type=float, default=1, help='Capacity factor')
    parser.add_argument('--use_residual', action='store_true', default=False, help='Use residual connections')
    parser.add_argument('--router_aux_loss_coef', type=float, default=0.01, help='Router auxiliary loss coefficient')
    parser.add_argument('--eval_capacity_factor', type=float, default=2, help='Evaluation capacity factor')
    parser.add_argument('--moe_layers_idx', type=str, default=None, help='Indices of layers to apply MOE')
    parser.add_argument('--min_capacity', type=int, default=0, help='Minimum capacity')
    parser.add_argument('--ep_size', type=int, default=1, help='EP size')
    parser.add_argument('--expert_pretrained_path', type=str, default=None, help='path of different MOE')
    parser.add_argument('--return_gating_logit', action='store_true', default=False, help='path of different MOE')

    return parser.parse_args(args)


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


class HookTool:
    def __init__(self):
        self.fea = None
    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out.detach().cpu()

def get_gating_logit_by_hook(model):
    fea_hooks = []
    for n, m in model.named_modules():
        if 'wg' in n and isinstance(m, torch.nn.Linear):
            print(n, m, 'match!!!!!!!!!!!!!!!!!!!!!!!!!')
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)
    return fea_hooks

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def main(args):
    args = parse_args(args)
    deepspeed.init_distributed(dist_backend='nccl')
    set_seed(42)
    if args.local_rank == 0:
        test_randomness()
        print('val_data_path is', args.val_data_path)
    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=True,
    )
    args.seg_token_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]



    conversation_lib.default_conversation = conversation_lib.conv_templates['v1']

    model_args = {
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "iou_loss_weight": args.bce_loss_weight,
        "focal_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "pretrain_mm_mlp_adapter": args.pretrain_mm_mlp_adapter,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half


    model_args = vars(args)
    model_args['test_only'] = True
    if args.moe_enable:
        model = MedPLIBForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, ignore_mismatched_sizes=True,
        **model_args
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
    model.resize_token_embeddings(len(tokenizer))


    vision_tower = model.get_model().get_vision_tower()
    # vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    if not args.cpu_only:
        model.to(dtype=torch_dtype, device=args.local_rank)

    if args.local_rank == 0:
        for name, param in model.named_parameters():
            param.requires_grad = False
            # print(f"Parameter Name: {name}, Shape: {param.shape}, grad: {param.requires_grad}")
            # print(param)
    # --------------show trainable params---------------
    def count_parameters(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params

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
                "image_processor": vision_tower.image_processor
                }

    data_args = types.SimpleNamespace(**data_args)

    val_dataset = LazySupervisedDataset(args.val_data_path, tokenizer, data_args, args.sam_img_size)

    if args.eval_vqa:
        val_indices = get_chunk(range(len(val_dataset)), args.num_chunks, args.chunk_idx)
        val_dataset = Subset(val_dataset, val_indices)
        


    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        # val_sampler = torch.utils.data.distributed.DistributedSampler(
        #     val_dataset, shuffle=False, drop_last=False
        # )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            # sampler=val_sampler,
            drop_last=False,
            collate_fn=partial(
                DataCollatorForSupervisedDataset,
                inference=True,
            ),
        )
    print(f"Validation dataset size: {len(val_dataset)}")

    if args.return_gating_logit:
        print(model)
        fea_hooks = get_gating_logit_by_hook(model)
    else:
        fea_hooks = None

    model.eval()
    if args.eval_seg:
        validate_seg(val_loader, model, 0, args, tokenizer, fea_hooks=fea_hooks)
        exit()
    if args.eval_vqa:
        validate_vqa(val_loader, model, 0, args, tokenizer, fea_hooks=fea_hooks)
        exit()



def calculate_iou(prediction_mask, ground_truth_mask):
    prediction_mask = prediction_mask.bool()
    ground_truth_mask = ground_truth_mask.bool()

    intersection = torch.logical_and(prediction_mask, ground_truth_mask)
    union = torch.logical_or(prediction_mask, ground_truth_mask)

    intersection_pixels = torch.sum(intersection)
    union_pixels = torch.sum(union)

    if union_pixels == 0:
        iou = 0
    else:
        iou = intersection_pixels.float() / union_pixels.float()

    return iou.item()


def save_binary_image(tensor_image, save_path):
    # Convert tensor to numpy array
    numpy_image = tensor_image.numpy()
    # Multiply binary image by 255
    numpy_image *= 255
    # Convert numpy array to uint8 type
    numpy_image = numpy_image.astype('uint8')
    # Save image using OpenCV
    cv2.imwrite(save_path, numpy_image)

def vis_overlay_masks(original_image_path, prediction_mask, ground_truth_mask, save_path):
    # Read the original image
    original_image = cv2.imread(original_image_path)
    
    # Convert original image and masks to RGB format
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    prediction_mask = prediction_mask.squeeze(0).cpu().numpy() * 255
    ground_truth_mask = ground_truth_mask.squeeze(0).cpu().numpy() * 255
    
    # Create semi-transparent light blue color (for overlay)
    # overlay_color = np.array([255, 0, 0], dtype=np.uint8)
    overlay_color = np.array([118, 158, 224], dtype=np.uint8)
    prediction_overlay = np.zeros_like(original_image)
    ground_truth_overlay = np.zeros_like(original_image)
    
    # Overlay semi-transparent color on masks
    prediction_overlay[prediction_mask > 0] = overlay_color
    ground_truth_overlay[ground_truth_mask > 0] = overlay_color
    
    # Merge original image and masks
    prediction_overlay_image = cv2.addWeighted(original_image, 0.5, prediction_overlay, 0.9, 0)
    ground_truth_overlay_image = cv2.addWeighted(original_image, 0.5, ground_truth_overlay, 0.9, 0)
    
    # prediction_overlay_image = prediction_overlay
    # ground_truth_overlay_image = ground_truth_overlay
    
    # Concatenate images horizontally
    combined_image = np.concatenate([original_image, ground_truth_overlay_image, prediction_overlay_image], axis=1)
    
    # Save the concatenated image
    cv2.imwrite(save_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def validate_vqa(val_loader, model_engine, epoch, args, tokenizer, fea_hooks=None):

    model_name = args.version.split('/')[-2]
    file_name = args.val_data_path.split('/')[-2]
    # answers_file = os.path.join('./runs', model_name,'infer_res', file_name + '.jsonl')

    answers_file = args.answers_file
    
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")

    if fea_hooks is not None:
        all_gating_logits = {}
    pbar = tqdm.tqdm(val_loader)
    idx = 0
    for input_dict in pbar:
    # for input_dict in val_loader:
        
        # if idx > 20:
        #     break

        if not args.cpu_only:
            torch.cuda.empty_cache()
            input_dict = dict_to_cuda(input_dict)

        if args.precision == "fp16":
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images_clip"] = input_dict["images_clip"].float()

        indices = (input_dict['input_ids'] == 29901).nonzero(as_tuple=True)
        input_ids = input_dict['input_ids'][:, :indices[1][-1]+1]
        attention_mask = input_dict['attention_mask'][:, :indices[1][-1]+1]
        with torch.no_grad():
            output_ids = model_engine.generate(
                input_ids,
                images=input_dict['images_clip'],
                attention_mask=attention_mask,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        if fea_hooks is not None:
            # import ipdb
            # ipdb.set_trace()
            image_tensor = input_dict['images_clip']
            all_gating_logits[idx] = dict(gating_logit=[i.fea for i in fea_hooks],
                                          images=image_tensor if image_tensor is None else image_tensor.detach().cpu(),
                                          input_ids=input_ids.detach().cpu(),
                                          output_ids=output_ids.detach().cpu())
            print(input_ids.shape, output_ids.shape, fea_hooks[0].fea.shape, image_tensor.shape if image_tensor is not None else [])
            # assert fea_hooks[0].fea.shape[0] + 1 == output_ids.shape[1] + 575
            print('The number of hooks is:', len(fea_hooks))


        input_token_len = input_ids.shape[1]
        
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        # if outputs.endswith(stop_str):
        #     outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                "image_path": input_dict['image_paths'][0],
                                "prompt": input_dict['questions_list'][0][0],
                                "gt": input_dict['gts_list'][0][0],
                                "text": outputs,
                                "answer_id": ans_id,
                                "model_id": model_name,
                                "answer_type": args.answer_type,
                                # "inputs": str(input_ids),
                                # "outputs": str(output_ids),
                                "metadata": {}}) + "\n")
        ans_file.flush()
        idx += 1
    ans_file.close()
    
    if fea_hooks is not None:
        torch.save(all_gating_logits, os.path.join('./runs', model_name,'infer_res', file_name + '_gating.pt'))


def validate_seg(val_loader, model_engine, epoch, args, tokenizer, fea_hooks=None):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    iou_meter = AverageMeter("IoU", ":6.3f", Summary.SUM)
    dice_meter = AverageMeter("Dice", ":6.3f", Summary.SUM)
    modality_metric = {}

    if fea_hooks is not None:
        all_gating_logits = {}

    pbar = tqdm.tqdm(val_loader)
    idx = 0
    for input_dict in pbar:
        # if idx > 200:
        #     break
        modality = input_dict["image_paths"][0].split('/')[-1].split('_')[0]
        if modality not in modality_metric.keys():
            modality_metric[modality] = {}
            modality_metric[modality]['iou'] = []
            modality_metric[modality]['dice'] = []

        if not args.cpu_only:
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

        indices = (input_dict['input_ids'] == 29901).nonzero(as_tuple=True)
        input_ids = input_dict['input_ids'][:, :indices[1][-1]+1]
        attention_mask = input_dict['attention_mask'][:, :indices[1][-1]+1]
        with torch.no_grad():
            # output_dict = model_engine(**input_dict)
            output_ids, pred_masks = model_engine.evaluate(
            input_dict["images_clip"],
            input_dict["images"],
            input_ids,
            input_dict['resize_list'],
            input_dict['label_list'],
            max_new_tokens=1024,
            tokenizer=tokenizer,
            attention_mask=attention_mask
        )
        if fea_hooks is not None:

            # import ipdb
            # ipdb.set_trace()
            image_tensor = input_dict['images_clip']
            all_gating_logits[idx] = dict(gating_logit=[i.fea for i in fea_hooks],
                                          images=image_tensor if image_tensor is None else image_tensor.detach().cpu(),
                                          input_ids=input_ids.detach().cpu(),
                                          output_ids=output_ids.detach().cpu())
            print(input_ids.shape, output_ids.shape, fea_hooks[0].fea.shape, image_tensor.shape if image_tensor is not None else [])
            # assert fea_hooks[0].fea.shape[0] + 1 == output_ids.shape[1] + 575
            print('The number of hooks is:', len(fea_hooks))



        # pred_masks = output_dict["pred_masks"]
        # masks_list = output_dict["gt_masks"][0].int().unsqueeze(0)
        masks_list = input_dict["masks_list"][0].int().unsqueeze(0)
        if len(pred_masks) == 0:
            acc_iou = 0.0
            iou = 0.0
            intersection = 0.0
            union = 0.0
        else:
            output_list = (torch.sigmoid(pred_masks[0]) > 0.1).int()
            # assert len(pred_masks) == 1
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                mask_i = mask_i.unsqueeze(0)
                output_i = output_i.unsqueeze(0)
                # intersection_i, union_i, _ = intersectionAndUnionGPU(
                #     output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                # )
                # intersection += intersection_i
                # union += union_i
                # acc_iou += intersection_i / (union_i + 1e-5)
                # acc_iou[union_i == 0] += 1.0  # no-object target
                iou = calculate_iou(output_i, mask_i)
            # intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            # acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        
        # intersection_meter.update(intersection), union_meter.update(
        #     union
        # ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

        iou_meter.update(iou)
        dice = 2*iou/(1+iou)
        dice_meter.update(dice)

        modality_metric[modality]['iou'].append(iou)
        modality_metric[modality]['dice'].append(dice)

        new_description = f'iou={iou}.'
        pbar.set_description(new_description)
        idx += 1
        if args.vis_mask:
            model_name = args.version.split('/')[-2]
            test_file_name = args.val_data_path.split('/')[-2]
            img_name = os.path.basename(input_dict['image_paths'][0])[:-4]
            save_path = os.path.join('./runs', model_name,'infer_res', test_file_name, img_name+'iou'+str(round(iou,4))+'.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vis_overlay_masks(input_dict['image_paths'][0], output_i, mask_i, save_path)


    model_name = args.version.split('/')[-2]
    file_name = args.val_data_path.split('/')[-2]
    if fea_hooks is not None:
        save_path = os.path.join('./runs', model_name,'infer_res', file_name + '_gating_seg.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(all_gating_logits, save_path)

    # intersection_meter.all_reduce()
    # union_meter.all_reduce()
    # acc_iou_meter.all_reduce()
    # iou_meter.all_reduce()
    # dice_meter.all_reduce()

    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # ciou = iou_class[1]
    # giou = acc_iou_meter.avg[1]
    miou = iou_meter.avg
    mDice = dice_meter.avg

    # print("giou: {:.6f}, ciou: {:.6f}".format(giou, ciou))
    print("miou: {:.6f}, mDice: {:.6f}".format(miou, mDice))

    modality_metric_res = {}
    for modality in modality_metric.keys():
        modality_metric_res[modality] = {}
        modality_metric_res[modality]['iou'] = round(sum(modality_metric[modality]['iou'])/len(modality_metric[modality]['iou']), 6)
        modality_metric_res[modality]['dice'] = round(sum(modality_metric[modality]['dice'])/len(modality_metric[modality]['dice']), 6)
    print(modality_metric_res)
    return miou, mDice


if __name__ == "__main__":
    main(sys.argv[1:])
