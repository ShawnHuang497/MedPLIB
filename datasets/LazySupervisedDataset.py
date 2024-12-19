
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import json
import copy
import os
import re

from torch.utils.data import Dataset
import transformers
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import random
import time
import torchvision
import torchvision

from utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, REGION_TOKEN_INDEX
from model.medplib import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    """
    Preprocess multimodal data.

    Args:
        sources (Sequence[str]): A sequence of strings representing the raw multimodal data.
        data_args (DataArguments): A data arguments object containing the necessary parameters for multimodal data processing.

    Returns:
        Dict: The preprocessed multimodal data in the form of a dictionary.

    """
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in str(sentence['value']):
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
                replace_token = DEFAULT_IMAGE_TOKEN
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources



def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Preprocess the input data for the model.

    Args:
        sources (List[Dict[str, str]]): A list of dictionaries, each representing a conversation turn with keys "from" and "value".
        tokenizer (transformers.PreTrainedTokenizer): A pre-trained tokenizer object from the transformers library.
        has_image (bool, optional): Whether the input data contains images. Defaults to False.

    Returns:
        Dict: A dictionary containing preprocessed data, including:
            - input_ids (torch.Tensor): Tokenized input IDs.
            - labels (torch.Tensor): Labels for masked targets.
            - conversations (List[str]): The original conversations.
            - question (List[str]): Extracted questions from the conversations.
            - gt (List[str]): Extracted ground truth responses from the conversations.

    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    question = []
    gt = []
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            # record the question and answer
            if sentence['from'] == 'human':
                question.append(sentence['value'].replace('<im_start><image><im_end>\n', ''))
            else:
                gt.append(sentence['value'])

            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print('tokenization mismatch, the question is', question)
                print('tokenization mismatch, the conversations is', conversations)
    return dict(
        input_ids=input_ids,
        labels=targets,
        conversations=conversations,
        question=question,
        gt=gt,
    )

def process_mask(mask):
    mask_tensor = torch.tensor(mask, dtype=torch.float)
    
    return mask_tensor

def extract_masks_fun(source, mask_root_path, pattern=r'<mask>(.*?)</mask>'):
    """
    Extract masks from the source.

    Args:
        source (dict): A dictionary containing conversation data.
        mask_root_path (str): The root path where the masks are stored.
        pattern (str, optional): The regular expression pattern to extract mask names. Defaults to r'<mask>(.*?)</mask>'.

    Returns:
        Tuple[List[np.ndarray], List[dict]]: A tuple containing a list of extracted masks as numpy arrays and a list containing the modified source dictionary.

    """
    extract_masks= []
    for idx, item in enumerate(source['conversations']):
        mask_name_lst = re.findall(pattern, str(item['value']))
        if mask_name_lst:
            assert len(mask_name_lst) == 1, "Only one mask is supported in one turns."
            mask_path = os.path.join(mask_root_path, mask_name_lst[0])
            mask = Image.open(mask_path).convert('L')
            mask_np = np.array(mask)

            mask_np[mask_np >= 1] = 1

            extract_masks.append(mask_np)
            if '</mask>' in pattern:
                assert '<SEG>' in item['value'], "SEG token is required in the answer when exist mask."
                source['conversations'][idx]['value'] = item['value'].replace(f'<mask>{mask_name_lst[0]}</mask>', '')
            elif '</region>' in pattern:
                source['conversations'][idx]['value'] = item['value'].replace(f'{mask_name_lst[0]}', '')
            else:
                print('extract mask path error.')

    return extract_masks, [source]

def generate_sub_connected_component(component, min_area, max_area, min_thresh=1000):
    # Calculate the area of the current connected component
    component_area = np.sum(component == 1)
    # Randomly select the ratio of the sub-connected component's area
    target_area = 0
    if component_area < min_thresh:
        # print('This component_area is too small', component_area)
        return component

    while target_area // min_thresh < 1:
        target_ratio = random.uniform(min_area, max_area)  
        # Calculate the target area of the sub-connected component
        target_area = int(component_area * target_ratio)
    
    # Generate a new sub-connected component within the current connected component
    sub_component = np.zeros_like(component)
    
    # Randomly select a starting point
    row, col = np.where(component == 1)
    start_point = random.choice(list(zip(row, col)))
    
    stack = [start_point]
    while len(stack) > 0:
        current_point = stack.pop()
        sub_component[current_point] = 1
        
        # Check if the area of the sub-connected component reaches the target area
        if np.sum(sub_component == 1) >= target_area:
            break
        
        # Randomly select a neighbor around the current point as the next point
        neighbors = [(current_point[0] + dy, current_point[1] + dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if 0 <= neighbor[0] < component.shape[0] and 0 <= neighbor[1] < component.shape[1] and component[neighbor] == 1 and sub_component[neighbor] == 0:
                stack.append(neighbor)
    
    return sub_component

def generate_mask_with_sub_component(masks, min_area=0.4, max_area=1.0, min_thresh=1000):
    sub_components = []
    is_valid = False
    for mask in masks:
        mask = np.array(mask)
        if np.sum(mask) > 0:
        # Get the connected components
            num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
            
            # Randomly select a connected component
            label_values = np.unique(labels)[1:]  # Remove background label 0

            max_area = 0
            max_area_label = 0
            for label_value in label_values:
                area = np.sum(labels == label_value)
                if area > max_area:
                    max_area = area
                    max_area_label = label_value
                    # print(max_area, 'max_area')
                is_valid = True
                # selected_label = random.choice(label_values)
                selected_label = max_area_label
                
                # Get the current connected component
                current_component = np.where(labels == selected_label, 1, 0)
                
                # Generate a sub-connected component
                sub_component = generate_sub_connected_component(current_component, min_area=min_area, max_area=max_area, min_thresh=min_thresh)
        else:
            is_valid = False
            sub_component = np.ones((336,336))

    
        sub_components.append(sub_component)
    return sub_components, is_valid




def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    i = 0
    element1 = tokenizer("<region>", add_special_tokens=False).input_ids[0]
    element2 = tokenizer("</region>", add_special_tokens=False).input_ids[0]
    while i < len(input_ids) - 1:
        if input_ids[i] == element1 and input_ids[i + 1] == element2:
            input_ids.insert(i + 1, REGION_TOKEN_INDEX)
            i += 1  
        i += 1

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    # for sam
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    # for clip
    clip_pixel_mean = (torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)*255).clamp(0, 255).to(torch.int)
    clip_pixel_std = (torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)*255).clamp(0, 255).to(torch.int)

    ignore_label = 255

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 image_size=1024):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.sam_img_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.clip_img_size = 336
        self.transform_clip = ResizeLongestSide(self.clip_img_size)


        self.list_data_dict = []
        for item in list_data_dict:
            for idx, conv_ in enumerate(item['conversations']):
                if not isinstance(item['conversations'][idx]['value'], str):
                    item['conversations'][idx]['value'] = str(item['conversations'][idx]['value'])
            self.list_data_dict.append(item)
            

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    def pad_tensor_channelwise(self, x, pad_h, pad_w, pad_values, is_mask=False):
        """
        Pad a 3-channel image tensor with different padding values for each channel,
        considering total padding length and odd padding size.

        Parameters:
        x (torch.Tensor): Input image tensor of shape (3, h, w).
        pad_h (int): Total padding size for the height.
        pad_w (int): Total padding size for the width.
        pad_values (tuple): A tuple of three elements specifying the padding value for each channel.

        Returns:
        torch.Tensor: Padded image tensor.
        """

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if is_mask:
            assert len(pad_values) == 1, "pad_values must have 1 elements, one for each channel."
            padded_tensor = torch.empty((x.shape[0] + pad_h, x.shape[1] + pad_w), dtype=x.dtype)
            padded_tensor[:, :] = pad_values[0]
            padded_tensor[pad_top:pad_top+x.shape[0], pad_left:pad_left+x.shape[1]] = x
        else:
            assert len(pad_values) == 3, "pad_values must have three elements, one for each channel."
            padded_tensor = torch.empty((3, x.shape[1] + pad_h, x.shape[2] + pad_w), dtype=x.dtype)
            for i in range(3):
                padded_tensor[i, :, :] = pad_values[i]
            padded_tensor[:, pad_top:pad_top+x.shape[1], pad_left:pad_left+x.shape[2]] = x

        return padded_tensor


    def preprocess(self, x: torch.Tensor, image_size: int, normalize: bool=True, is_mask: bool=False) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        if normalize:
            x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = image_size - h
        padw = image_size - w
        if is_mask:
            x = self.pad_tensor_channelwise(x, padh, padw, torch.zeros(1), is_mask=True)
        else:
            # for sam. pad after normalize
            if normalize:
                x = self.pad_tensor_channelwise(x, padh, padw, torch.zeros(3))
                # x = x * self.pixel_std + self.pixel_mean

            # for clip. pad before normalize
            else:
                x = self.pad_tensor_channelwise(x, padh, padw, self.clip_pixel_mean)

        return x


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        answer_type = sources.get('answer_type', None)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        # for segmentation
        masks, sources = extract_masks_fun(sources[0], self.data_args.image_folder, pattern=r'<mask>(.*?)</mask>')
        # for region prompt
        region_masks, sources = extract_masks_fun(sources[0], self.data_args.image_folder, pattern=r'<region>(.*?)</region>')
 
        region_masks = [self.transform_clip.apply_image(region_mask.astype(np.uint8)) for region_mask in region_masks]
        region_masks = [self.preprocess(torch.from_numpy(region_mask).contiguous(), self.clip_img_size, normalize=False, is_mask=True) for region_mask in region_masks]
        start = time.time()
        region_masks = [cv2.resize(np.array(region_mask), None, fx=1/14, fy=1/14, interpolation=cv2.INTER_NEAREST) for region_mask in region_masks]
        region_masks, is_valid_region = generate_mask_with_sub_component(region_masks, min_area=0.2, max_area=1, min_thresh=10)
        # print('generate_mask_with_sub_component', time.time() - start)


        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            if os.path.exists(image_file):
                image_path = image_file
            elif 'llavamed' in image_file:
                image_path = os.path.join('/'.join(image_folder.split('/')[:-1]), image_file)
            else:
                image_path = os.path.join(image_folder, image_file)

            #------------ preprocess image for sam ------------
            assert os.path.exists(image_path), f'{image_path} dose not exist'
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resize = self.transform.apply_image(image_rgb)
            resize = image_resize.shape[:2]
            image_sam = self.preprocess(torch.from_numpy(image_resize).permute(2, 0, 1).contiguous(), self.sam_img_size)


            #------------preprocess image for clip  ------------
            # c, h, w -> h, w, c
            image_clip = self.transform_clip.apply_image(image_rgb)
            #c, h, w
            image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_img_size, normalize=False)
            # torchvision.transforms.functional.to_pil_image(image_clip.byte()).save('/root/paddlejob/workspace/env_run/output/LISA/image_clip_0.png')
            # preprocess image for clip
            if self.data_args.image_aspect_ratio == 'pad':
                #c, h, w
                image_clip = processor.preprocess(image_clip, return_tensors='pt')['pixel_values'][0]
                # image_clip = processor.preprocess(image_rgb, return_tensors='pt')['pixel_values'][0]
            else:
                image_clip = processor.preprocess(self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), False), return_tensors='pt')['pixel_values'][0]
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))

        masks = [process_mask(mask) for mask in masks]
        region_masks = [process_mask(mask).unsqueeze(0) for mask in region_masks]

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             conversations=data_dict["conversations"],
                             question=data_dict["question"],
                             gt=data_dict["gt"],
                             )

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image_clip'] = image_clip
            data_dict['masks'] = masks
            data_dict['region_masks'] = region_masks
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image_clip'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        data_dict['image_sam'] = image_sam
        data_dict["image_path"] = image_path



        data_dict['inference'] = False
        data_dict['tokenizer'] = self.tokenizer

        data_dict['answer_type'] = answer_type

        # for sam restoring mask
        if len(masks) > 0:
            label = [torch.ones(masks[0].shape[0], masks[0].shape[1]) * self.ignore_label] * len(masks)
            data_dict['label'] = label
            data_dict['resize'] = [resize] * len(masks)

        # do not cal loss, if region mask is invalid
        if len(region_masks)> 0 and not is_valid_region:
            data_dict['labels'] = torch.zeros_like(data_dict["labels"])
            data_dict['labels'][...] = -100
            
            tmp_region = torch.zeros(1, 336, 336)
            tmp_region[:,:40,:40] = 1
            # print(torch.sum(tmp_region))
            data_dict['region_masks'] = [tmp_region]
            # print(f'{image_path} is invalid')

        return data_dict
        
