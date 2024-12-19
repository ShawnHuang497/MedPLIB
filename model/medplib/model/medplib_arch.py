#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

# from medplib.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX,
                         IMAGE_TOKEN_INDEX, REGION_TOKEN_INDEX)

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from ...rp_sampler import GeoRegionSampler

def rand_sample(x, max_len):
    if x.shape[0] <= max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[0])[:max_len]
    return x[rand_idx, :]

def point_sample(input, point_coords, return_dtype, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    # output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    output = F.grid_sample(input.float(), (2.0 * point_coords - 1.0).float(), **kwargs)
    output = output.to(return_dtype)
    if add_dim:
        output = output.squeeze(3)
    return output


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        # if hasattr(config, "mm_vision_tower"):
        self.vision_tower = build_vision_tower(config, delay_load=False)
        # self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        self.mm_projector = build_vision_projector(config)

        # if config.region_fea_adapter:
        self.region_fea_adapter = nn.Linear(config.mm_hidden_size, config.hidden_size)
        print('LlavaMetaModel config', config)

        # if config.region_geo_sampler:
        #     # pdb.set_trace()
        #     self.region_geo_sampler = GeoRegionSampler(input_dim=config.mm_hidden_size,
        #                                                output_dim=config.hidden_size,
        #                                                num_init_point=config.max_sample_point,
        #                                                num_sub_point=[128, 32],
        #                                                num_neighbor=[24, 24],
        #                                                pooler_mode=config.sampler_pooler_mode
        #                                                )
        self.max_sample_point = config.max_sample_point
    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        print('load vision_tower successfully!!!')
        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature


        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )
            print('load projector successfully!!!')
            

class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, region_flag=False, region_geo_sampler=False):
        image_features = self.get_model().get_vision_tower()(images)
        projected_image_features = self.get_model().mm_projector(image_features)

        if region_flag:
            if region_geo_sampler:
                new_region_feature_map = image_features
            else:
                new_region_feature_map = self.get_model().region_fea_adapter(image_features)
        else:
            new_region_feature_map = None

        return image_features, projected_image_features, new_region_feature_map

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, region_masks, valid_region_masks_bool
    ):
        if region_masks is None:
            region_flag = False
        else:
            if len(region_masks) > 0:
                region_flag = True
            else:
                region_flag = False

        region_geo_sampler = region_flag and getattr(self.config, 'region_geo_sampler', False)

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels
        if type(images) is list or images.ndim == 5:
            assert region_flag == False
            concat_images = torch.cat([image for image in images], dim=0)
            raw_image_features, image_features, region_feature_map = self.encode_images(concat_images, region_flag, region_geo_sampler)
            # image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            raw_image_features, image_features, region_feature_map = self.encode_images(images, region_flag, region_geo_sampler)
        if region_flag:
            valid_region_masks_bool = torch.tensor([any(item) for item in valid_region_masks_bool])
            region_feature_map = region_feature_map[valid_region_masks_bool]

            if region_geo_sampler:
                # pdb.set_trace()
                region_features = self.get_model().region_geo_sampler(region_feature_map, region_masks, 
                                                                      original_dtype=raw_image_features.dtype,
                                                                      return_dtype=image_features.dtype)
            else:
                region_features = self.extract_region_feature(region_feature_map, region_masks, 
                                                              original_dtype=raw_image_features.dtype,
                                                              return_dtype=image_features.dtype)
            # assert len(region_features) == len(input_ids)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = (
                    cur_input_embeds
                    + (
                        0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)
                    ).sum()
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if region_flag:
                    assert (cur_input_ids[:image_token_start] == REGION_TOKEN_INDEX).sum() == 0
                # If not use start-end token, pt ckpt saved only has mm projector.
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1 : image_token_start + 2]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                # print(type(cur_input_ids),cur_input_ids.shape)
                region_indices = (cur_input_ids == REGION_TOKEN_INDEX).nonzero(as_tuple=True)[0].tolist()
                cur_input_ids = cur_input_ids[cur_input_ids != REGION_TOKEN_INDEX]
                # print('region_indices',region_indices)

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    text_input_embeds = self.get_model().embed_tokens(cur_input_ids).detach()
                else:
                    text_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                if labels is not None:
                    cur_new_labels.append(cur_labels)

                # Add region feature into text feature embeddings.
                assert batch_idx+1 == cur_image_idx
                if region_flag and valid_region_masks_bool[batch_idx] is not None:
                    for idx,indice in enumerate(region_indices):
                        region_features_idx = torch.sum(valid_region_masks_bool[:batch_idx+1]).item() - 1
                        text_input_embeds = torch.cat((text_input_embeds[:indice], region_features[region_features_idx][idx].unsqueeze(0), text_input_embeds[indice:]))
                else:
                    if region_flag:
                        assert region_features[batch_idx] is None

                cur_new_input_embeds.append(text_input_embeds)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    # def initialize_vision_tokenizer(self, model_args, tokenizer):
    def initialize_vision_tokenizer(self, model_args, num_new_tokens):
        # if model_args.mm_use_im_patch_token:
        #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        #     self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            # num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            # self.resize_token_embeddings(len(tokenizer))

            # if num_new_tokens > 0:
            #     input_embeddings = self.get_input_embeddings().weight.data
            #     output_embeddings = self.get_output_embeddings().weight.data

            #     input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)
            #     output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)

            #     input_embeddings[-num_new_tokens:] = input_embeddings_avg
            #     output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def extract_region_feature(self, region_feature_map, region_masks, original_dtype, return_dtype):
        all_region_features = []
        assert len(region_feature_map) == len(region_masks), f'{len(region_feature_map)}, {len(region_masks)}'

        for region_feature_map_i, region_masks_list_i in zip(region_feature_map, region_masks):
            if len(region_masks_list_i) == 0:
                all_region_features.append(None)
            else:
                # (w, h)
                ori_image_wh = torch.tensor([region_masks_list_i[0].shape[0], region_masks_list_i[0].shape[1]], device=region_masks_list_i[0].device)[None,]
                # list of elements of shape [num_sample_point, 2]
                non_zero_pos = [rand_sample((m.nonzero()/ori_image_wh), self.get_model().max_sample_point) for m in region_masks_list_i]
                # [num_mask, num_sample_point(padded), 2]
                non_zero_pos = nn.utils.rnn.pad_sequence(non_zero_pos, padding_value=-1, batch_first=True)
                non_zero_pos_mask = ~(non_zero_pos.sum(dim=-1) < 0)
                # [HxW, C] -> [H, W, C] -> [C, H, W] -> [N, C, H, W]
                h = w = int(math.sqrt(region_feature_map_i.shape[0]))
                c = region_feature_map_i.shape[-1]
                dup_region_feature_map_i = region_feature_map_i.reshape(h, w, c).permute(2, 0, 1)
                dup_region_feature_map_i = dup_region_feature_map_i.unsqueeze(0).repeat(non_zero_pos.shape[0], 1, 1, 1)
                # [num_mask, C, H, W] x [num_mask, num_sample_point(padded), 2] -> [num_mask, C, num_sample_point(padded)]
                # F.grid_sample doesn't support BF16. Need to tranform into float32 then transform back.
                dup_region_feature_map_i_ori_type = dup_region_feature_map_i.to(original_dtype)
                # pdb.set_trace()
                region_feature_i = point_sample(dup_region_feature_map_i_ori_type, 
                                                non_zero_pos.flip(dims=(2,)).type(original_dtype), 
                                                return_dtype, 
                                                align_corners=True
                                                )
                region_feature_i = region_feature_i.to(dup_region_feature_map_i.dtype)
                # [num_mask, C]
                region_feature_i = torch.stack([x[m].mean(dim=0) for x, m in zip(region_feature_i.transpose(1,2), non_zero_pos_mask)]).nan_to_num()
                all_region_features.append(region_feature_i)
        
        return all_region_features