
from typing import Any, Dict, List, Tuple
from typing import Dict, Optional, Sequence, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .medplib.model.language_model.medplib_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
# from .segment_anything import build_sam_vit_h
from .segment_anything_med2d import build_sam_vit_b


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss

def dice_loss(inputs, targets, num_masks=0, eps=1e-6):
    """
    Calculate the Dice loss, which is a measure of similarity between two sets.
    This function is particularly useful for binary classification tasks, such as image segmentation.

    Args:
        inputs (torch.Tensor): A tensor containing the raw output predictions from the model,
                               which are typically logits that need to be sigmoid-transformed.
        targets (torch.Tensor): A tensor containing the ground truth labels, where each element
                                is 0 or 1.
        eps (float): A small epsilon value to avoid division by zero in the loss computation.

    Returns:
        torch.Tensor: The computed Dice loss.
    """
    # Apply sigmoid to the input to transform logits to probabilities
    inputs = torch.sigmoid(inputs)
    
    # Flatten inputs and targets to ensure the loss is computed element-wise
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    
    # Calculate the intersection and the union components of the Dice score
    intersection = (inputs * targets).sum(-1)
    union = inputs.sum(-1) + targets.sum(-1)
    
    # Compute Dice score
    dice_score = (2. * intersection + eps) / (union + eps)
    
    # Dice loss is 1 minus Dice score
    dice_loss = 1 - dice_score

    # Average the loss across the batch
    return dice_loss.mean()



def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        # self.config.train_mask_decoder = kwargs["train_mask_decoder"]
        # self.config.out_dim = kwargs["out_dim"]
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        # self.visual_model = build_sam_vit_h(self.vision_pretrained)
        self.visual_model = build_sam_vit_b(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        if 'pretrain_mm_mlp_adapter' not in kwargs:
            self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # print('0000config',config)
        # print('0000kwargs',kwargs)

        config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
        if not hasattr(config, "vision_tower"):
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14")
        else:
            config.mm_vision_tower = config.vision_tower
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.iou_loss_weight = kwargs.pop("iou_loss_weight", None)
        self.focal_loss_weight = kwargs.pop("focal_loss_weight", None)
        config.train_mask_decoder = kwargs.get("train_mask_decoder", True)
        config.out_dim = kwargs.get("out_dim", 256)

        self.seg_token_idx = kwargs.pop("seg_token_idx")

        config.region_fea_adapter =  kwargs.get("region_fea_adapter")
        config.region_geo_sampler =  kwargs.get("region_geo_sampler")
        config.max_sample_point =  kwargs.get("max_sample_point")
        config.sampler_pooler_mode =  kwargs.get("sampler_pooler_mode")

        super().__init__(config)
        print('LISAForCausalLM config', config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.iou_loss_fn = MaskIoULoss()
        self.focal_loss_fn = FocalLoss()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def is_empty_tensor(self, tensor):
        return tensor.numel() == 0



    def expand_embedding(self, image_embeddings, valid_mask_bool):
        if len(valid_mask_bool) == 0:
            return image_embeddings
        expand_factors = [len(mask) if mask else 0 for mask in valid_mask_bool]
        expanded_tensors = []  
        for i, factor in enumerate(expand_factors):
            if factor > 0:
                expanded_tensor = torch.index_select(image_embeddings, 0, torch.tensor([i]).cuda())  
                expanded_tensor = expanded_tensor.expand(factor, -1, -1, -1) 
                expanded_tensors.append(expanded_tensor)
        image_embeddings_expanded = torch.cat(expanded_tensors, dim=0)
        return image_embeddings_expanded



    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        region_masks: List[torch.FloatTensor],
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        seg_flag: bool = True,
        valid_mask_bool: torch.BoolTensor = None,
        rp_flag: bool = False,
        valid_region_masks_bool: Optional[List[torch.BoolTensor]] = [],
        **kwargs,
    ):
        """
        Forward pass of the model.
        
        Args:
            images (torch.FloatTensor): Input images with shape (batch_size, channels, height, width).
            images_clip (torch.FloatTensor): Clipped input images with shape (batch_size, channels, clip_height, clip_width).
            input_ids (torch.LongTensor): Input token IDs with shape (batch_size * question_instance_number, sequence_length).
            labels (torch.LongTensor): Labels with shape (batch_size * question_instance_number, sequence_length).
            attention_masks (torch.LongTensor): Attention masks with shape (batch_size * question_instance_number, sequence_length).
            offset (torch.LongTensor): Offsets of each image in the batch with shape (batch_size + 1).
            masks_list (List[torch.FloatTensor]): List of ground truth masks with shape (batch_size, 3, height, width).
            label_list (List[torch.Tensor]): List of labels with shape (batch_size, height, width).
            resize_list (List[tuple]): List of tuples containing the resized input size with shape (batch_size).
            inference (bool, optional): Whether the model is in inference mode. Defaults to False.
            **kwargs: Additional keyword arguments.
        
        Returns:
            dict: A dictionary containing the model output, including loss, cross-entropy loss, mask BCE loss, mask Dice loss, and mask loss.
        
        """
        if seg_flag:
            # try:
            # [b, 256, 64, 64]
            image_embeddings = self.get_visual_embs(images)
            # except:
            #     print('error image shape is ', images.shape)
            #     print('error image path is ', kwargs['image_paths'])
            image_embeddings = self.expand_embedding(image_embeddings, valid_mask_bool)
            batch_size = image_embeddings.shape[0]
            # assert batch_size == len(offset) - 1

            # B*q_num, sequence_length-1
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
            # B*q_num, sequence_length+1
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
                ],
                dim=1,
            )
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            # B*q_num, 255+sequence_length
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], 575)).bool().cuda(), seg_token_mask],
                dim=1,
            )
        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            region_masks=region_masks,
            valid_region_masks_bool=valid_region_masks_bool,
        )
        # 33, B*q_num, N, 4096
        output_hidden_states = output.hidden_states
        ce_loss = output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        if not seg_flag:
            return {
                "loss": ce_loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": torch.zeros_like(ce_loss),
                "mask_dice_loss": torch.zeros_like(ce_loss),
                "mask_loss": torch.zeros_like(ce_loss),
                "unscale_mask_bce_loss": torch.zeros_like(ce_loss),
                "unscale_mask_dice_loss": torch.zeros_like(ce_loss),
                "unscale_mask_loss": torch.zeros_like(ce_loss),
                "unscale_mask_iou_loss": torch.zeros_like(ce_loss),
                "unscale_mask_focal_loss": torch.zeros_like(ce_loss),
            }


        # hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1

        # laryers, B*q_num, N, 4096 --> B*q_num, N, 256
        last_hidden_state = self.model.text_hidden_fcs[0](output_hidden_states[-1])

        if inference:
            last_hidden_state = torch.unsqueeze(last_hidden_state, dim=0)
        # B*q_num, 255+sequence_length, 256  [B*q_num, 255+sequence_length]  --> B*q_num, 256
        pred_embeddings = last_hidden_state[seg_token_mask]

        # B*q_num
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]


        multimask_output = False
        pred_masks = []
        pred_ious = []
        for i in range(len(pred_embeddings)):
            #  pred_embeddings[i].shape, pred_embeddings[i].unsqueeze(1).shape torch.Size([3, 256]) torch.Size([3, 1, 256])
            # sparse_embeddings: [q_num, 1, 256] dense_embeddings: [q_num, 256, 64, 64]
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(0).unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            # low_res_masks: [3, 1, 256, 256], resize_list[i]: [685, 1024], label_list[i]: [h, w]
            # pred_mask: [1, 1, h, w]
            # pred_mask = self.model.visual_model.postprocess_masks(
            pred_mask = self.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])
            # iou_predictions: [1,1]
            pred_ious.append(iou_predictions[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        mask_bce_loss = 0
        mask_dice_loss = 0
        mask_iou_loss = 0
        mask_focal_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx].unsqueeze(0)
            pred_mask = pred_masks[batch_idx]
            # print(gt_mask.dtype, pred_mask.dtype)

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_iou_loss += self.iou_loss_fn(pred_mask, gt_mask, pred_ious[batch_idx]) * gt_mask.shape[0]
            mask_focal_loss += self.focal_loss_fn(pred_mask, gt_mask) * gt_mask.shape[0]
            
            num_masks += gt_mask.shape[0]

        unscale_mask_bce_loss= mask_bce_loss / (num_masks + 1e-8)
        mask_bce_loss = self.bce_loss_weight * unscale_mask_bce_loss
        unscale_mask_dice_loss = mask_dice_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * unscale_mask_dice_loss

        unscale_mask_iou_loss = mask_iou_loss / (num_masks + 1e-8)
        unscale_mask_focal_loss = mask_focal_loss / (num_masks + 1e-8)
        mask_iou_loss = self.iou_loss_weight * unscale_mask_iou_loss
        mask_focal_loss = self.focal_loss_weight * unscale_mask_focal_loss

        unscale_mask_loss = unscale_mask_bce_loss + unscale_mask_dice_loss + unscale_mask_iou_loss + unscale_mask_focal_loss

        mask_loss = mask_bce_loss + mask_dice_loss + mask_iou_loss + mask_focal_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "unscale_mask_bce_loss": unscale_mask_bce_loss,
            "unscale_mask_dice_loss": unscale_mask_dice_loss,
            "unscale_mask_loss": unscale_mask_loss,
            "unscale_mask_iou_loss": unscale_mask_iou_loss,
            "unscale_mask_focal_loss": unscale_mask_focal_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 575)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            # B*q_num, N, 256 --> B*q_num, 255+sequence_length, 256
            # last_hidden_state = last_hidden_state[:, :seg_token_mask.shape[1], :]
            pred_embeddings = last_hidden_state[seg_token_mask]

            print(pred_embeddings.shape)
            if pred_embeddings.shape[0] > 1:
                pred_embeddings = pred_embeddings[:1, :]
            elif pred_embeddings.shape[0] == 0:
                pred_embeddings = last_hidden_state[:1, -2:-1, :].squeeze(1)

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                # print(pred_embeddings[i].shape)
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(0).unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                # pred_mask = self.model.visual_model.postprocess_masks(
                pred_mask = self.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i].shape,
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks

    def postprocess_masks(self, masks: torch.Tensor, input_size: Tuple[int, ...],original_size: Tuple[int, ...]):
        """
        Removes padding from a padded tensor and retrieves the original tensor shape.

        Returns:
        torch.Tensor: The original image tensor with shape (b, 3, original_height, original_width).
        """
        if len(masks.shape) == 3:
            masks = masks.unsqueeze(0)
        pad_h = masks.shape[-2] - input_size[0]
        pad_w = masks.shape[-1] - input_size[1]
        # cal the pad 
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        original_height = masks.shape[-2] - pad_h
        original_width = masks.shape[-1] - pad_w

        # get origin area and resize to original size
        masks = masks[:, :, pad_top:pad_top+original_height, pad_left:pad_left+original_width]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks