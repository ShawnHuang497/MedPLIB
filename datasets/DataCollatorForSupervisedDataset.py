
import transformers
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from utils.utils import IGNORE_INDEX



def DataCollatorForSupervisedDataset(list_data_dict: Sequence[Dict], inference: bool = False) -> Dict[str, torch.Tensor]:
    tokenizer = list_data_dict[0]['tokenizer']
    input_ids, labels = tuple([instance[key] for instance in list_data_dict]
                                for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
    input_ids = input_ids[:, :tokenizer.model_max_length]
    labels = labels[:, :tokenizer.model_max_length]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

    seg_flag = False
    max_mask_nums = max([len(instance['masks']) for instance in list_data_dict])
    #### fullfill mask instances according to the max_mask_nums in this batch
    masks = []
    label_list = []
    valid_mask_bool = []
    resize_list = []
    if max_mask_nums > 0:
        seg_flag = True
        for idx, instance in enumerate(list_data_dict):
            if len(instance['masks'])>0:
                masks.extend(instance['masks'])
                label_list.extend(instance['label'])
                resize_list.extend(instance['resize'])
                valid_mask_bool.append([True]* len(instance['masks']))
            else:
                valid_mask_bool.append([])

    batch['masks'] = masks
    batch['valid_mask_bool'] = valid_mask_bool
    batch['label_list'] = label_list
    batch['resize_list'] = resize_list


    region_masks = []
    valid_region_masks_bool = []
    max_region_masks_nums = max([len(instance['region_masks']) for instance in list_data_dict])
    
    rp_flag = False
    if max_region_masks_nums > 0:
        rp_flag = True
        for idx, instance in enumerate(list_data_dict):
            if len(instance['region_masks'])>0:
                region_masks.extend(instance['region_masks'])
                valid_region_masks_bool.append([torch.ones(1).bool()]* len(instance['region_masks']))
            else:
                valid_region_masks_bool.append([torch.zeros(1).bool()])

    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    # resize_list = []
    # label_list = []
    questions_list = []
    gts_list = []
    sampled_classes_list = []
    offset_list = [0]
    answer_type_list = []
    cnt = 0
    for data_dict in list_data_dict:
        image_path_list.append(data_dict.get('image_path', None))
        images_list.append(data_dict.get('image_sam', None))
        images_clip_list.append(data_dict.get('image_clip', None))
        conversation_list.extend(data_dict.get('conversations', None))
        # label_list.append(data_dict.get('label', torch.tensor([])))
        # resize_list.append(data_dict.get('resize', None))
        questions_list.append(data_dict.get('question', None))
        gts_list.append(data_dict.get('gt', None))
        sampled_classes_list.append(data_dict.get('sampled_classes', None))
        cnt += len(data_dict.get('conversations', None))
        offset_list.append(cnt)
        answer_type_list.append(data_dict.get('answer_type', None))

    final_batch = {
            "image_paths": image_path_list,
            "images": torch.stack(images_list, dim=0),
            "images_clip": torch.stack(images_clip_list, dim=0),
            "input_ids": batch['input_ids'],
            "labels": batch['labels'],
            "attention_mask": batch['attention_mask'],
            "masks_list": batch['masks'],
            "label_list": batch['label_list'],
            "resize_list": resize_list,
            "offset": torch.LongTensor(offset_list),
            "questions_list": questions_list,
            "gts_list": gts_list,
            "sampled_classes_list": sampled_classes_list,
            "conversation_list": conversation_list,
            "seg_flag": seg_flag,
            "valid_mask_bool": batch.get('valid_mask_bool', []),
            "inference": inference,
            "answer_type_list": answer_type_list,
            "rp_flag": rp_flag,
            "region_masks": region_masks,
            "valid_region_masks_bool": valid_region_masks_bool,
        }
    return final_batch