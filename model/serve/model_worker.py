"""
A model worker executes the model.
Usage:

CUDA_VISIBLE_DEVICES=0 python -m model.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 \
    --worker http://localhost:40000 --model-path checkpoints/xxx \
    --multi-modal --add_region_feature

"""
import argparse
import asyncio
import json
import time
import threading
import uuid
import base64

import numpy as np
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
from functools import partial
import cv2

from model.medplib.constants import WORKER_HEART_BEAT_INTERVAL
from model.medplib.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from model.medplib.model.builder import load_pretrained_model
from model.medplib.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from model.medplib.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from utils.utils import REGION_TOKEN_INDEX, DEFAULT_REGION_REFER_TOKEN_0, DEFAULT_REGION_REFER_TOKEN_1

# from transformers import TextIteratorStreamer
from threading import Thread

from model.segment_anything.utils.transforms import ResizeLongestSide
import deepspeed
deepspeed.init_distributed(dist_backend='nccl',distributed_port=63999)

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"./serve_logs/model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None

DEFAULT_REGION_FEA_TOKEN = "<region_fea>"

def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:

    # for sam
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    # for clip
    clip_pixel_mean = (torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)*255).clamp(0, 255).to(torch.int)
    clip_pixel_std = (torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)*255).clamp(0, 255).to(torch.int)

    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, load_fp16,
                 keep_aspect_ratio,
                 num_gpus,
                 add_region_feature,
                 image_w,
                 image_h,
                 args):
        self.args = args
        self.image_w = image_w
        self.image_h = image_h
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.keep_aspect_ratio = keep_aspect_ratio
        self.add_region_feature = add_region_feature
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, load_fp16, device_map=args.device_map, vision_pretrained=args.vision_pretrained)
            
        
        self.is_multimodal = 'llava' in self.model_name.lower() or 'medplib' in self.model_name.lower()

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

        self.sam_img_size = 256
        self.transform = ResizeLongestSide(self.sam_img_size)
        self.clip_img_size = 336
        self.transform_clip = ResizeLongestSide(self.clip_img_size)
        self.precision = "bf16"
        self.seg_token_idx = self.tokenizer("<SEG>", add_special_tokens=False).input_ids[0]

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        print(url, self.worker_addr, self.get_status())

        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        try:
            r = requests.post(url, json=data, timeout=60)
            assert r.status_code == 200
        except Exception as e:
            logger.error(f"Register to controller failed: {e}")
            print('33'*20)
            raise e

    def send_heart_beat(self):
        # logger.info(f"Send heart beat. Models: {[self.model_name]}. "
        #             f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
        #             f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

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

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        image_w = self.image_w
        image_h = self.image_h
        prompt = params["prompt"]
        ori_prompt = prompt
        region_masks = params.get('region_masks', None)

        images = params.get("images", None)
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                # only support one image each time
                image_rgb = np.array([load_image_from_base64(image) for image in images][0])
                origin_shape = image_rgb.shape[:2]


                #------------ preprocess image for sam ------------
                image_resize = self.transform.apply_image(image_rgb)
                resize_shape = image_resize.shape[:2]
                image_sam = self.preprocess(torch.from_numpy(image_resize).permute(2, 0, 1).contiguous(), self.sam_img_size)


            
                #------------preprocess image for clip  ------------
                # c, h, w -> h, w, c
                image_clip = self.transform_clip.apply_image(image_rgb)
                #c, h, w
                image_clip = self.preprocess(torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(), self.clip_img_size, normalize=False)
                #c, h, w
                image_clip = image_processor.preprocess(image_clip, return_tensors='pt')['pixel_values'][0]

            else:
                images = None
        else:
            images = None

        # deal region prompt mask
        valid_region_masks_bool = []
        if region_masks is not None:
            region_masks = [np.array(region_mask_i) for region_mask_i in region_masks]
            region_masks = [self.transform_clip.apply_image(region_mask.astype(np.uint8)) for region_mask in region_masks]
            region_masks = [self.preprocess(torch.from_numpy(region_mask).contiguous(), self.clip_img_size, normalize=False, is_mask=True) for region_mask in region_masks]
            region_masks = [cv2.resize(np.array(region_mask), None, fx=1/14, fy=1/14, interpolation=cv2.INTER_NEAREST) for region_mask in region_masks]
            valid_region_masks_bool.append([torch.ones(1).bool()])
            logger.info("Add region_masks to image_args.")
            # for debug
            # cv2.imwrite('/root/paddlejob/workspace/env_run/output/huangxiaoshuang/MedPLIB/serve_images/resize_region.png', region_masks[0]*255)
        else:
            logger.info("No region_masks for this sample.")
            region_masks = None

        l_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)

        stop_idx = None
        if stop_str is not None:
            stop_idx = tokenizer(stop_str).input_ids
            if len(stop_idx) == 1:
                stop_idx = stop_idx[0]
            else:
                stop_idx = None

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
        output_ids = list(input_ids)

        # add region feat placeholder
        i = 0
        element1 = tokenizer("<region>", add_special_tokens=False).input_ids[0]
        element2 = tokenizer("</region>", add_special_tokens=False).input_ids[0]
        while i < len(input_ids) - 1:
            if input_ids[i] == element1 and input_ids[i + 1] == element2:
                input_ids.insert(i + 1, REGION_TOKEN_INDEX)
                i += 1  
            i += 1

        pred_ids = []

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        input_ids = torch.as_tensor([input_ids])

        past_key_values = None


        if self.precision == "fp16":
            image_clip = image_clip.unsqueeze(0).half()
            image_sam=image_sam.unsqueeze(0).half()
        elif self.precision == "bf16":
            image_clip = image_clip.unsqueeze(0).bfloat16()
            image_sam=image_sam.unsqueeze(0).bfloat16()
        elif self.precision == "fp32":
            image_clip = image_clip.unsqueeze(0).float()
            image_sam=image_sam.unsqueeze(0).float()
        elif self.precision == "int8":
            image_clip = image_clip.unsqueeze(0).int()
            image_sam=image_sam.unsqueeze(0).int()
        elif self.precision == "int4":
            image_clip = image_clip.unsqueeze(0).int()
            image_sam=image_sam.unsqueeze(0).int()
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")
        
        if self.args.device_map == 'cuda':
            image_clip = image_clip.to("cuda")
            image_sam=image_sam.to("cuda")
            input_ids = input_ids.to("cuda")
            region_masks = torch.tensor([region_masks]).cuda()
            
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        resize_list = [list(resize_shape)]
        original_size_list = [torch.ones(origin_shape)]


# ----------------------------------return all token once---------------------------------------------
        # output_ids, pred_masks = model.evaluate(
        #             images_clip=image_clip,
        #             images=image_sam,
        #             input_ids=input_ids,
        #             resize_list=[list(resize_shape)],
        #             original_size_list=[torch.ones(origin_shape)],
        #             region_masks=region_masks,
        #             valid_region_masks_bool=valid_region_masks_bool,
        #             max_new_tokens=max_new_tokens,
        #             tokenizer=tokenizer,
        #             attention_mask=input_ids.ne(tokenizer.pad_token_id),
        #             inference_demo=True,
        #             )

        # input_token_len = input_ids.shape[1]
        # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        # outputs = outputs.strip()
        # output = ori_prompt + outputs
        
        # if len(pred_masks) > 0:
        #     pred_mask = (torch.sigmoid(pred_masks[0]) > 0.1).int().squeeze(0).cpu().numpy()
        #     height, width = pred_mask.shape
        #     def encode_sparse(mask):
        #         non_zero_positions = np.transpose(np.nonzero(mask))
        #         encoded_data = non_zero_positions.tolist()
        #         return encoded_data
        #     encoded_mask = encode_sparse(pred_mask)
        # else:
        #     encoded_mask = []
        #     height = width = 0

        # ret = {
        #     "text": output,
        #     "mask": encoded_mask,
        #     "height": str(height),
        #     "width": str(width),
        #     "error_code": 0,
        # }

        # if past_key_values is not None:
        #     del past_key_values

        # yield json.dumps(ret).encode() + b"\0"

# ----------------------------------return token by token---------------------------------------------
        all_output_hidden_states = []
        for i in range(max_new_tokens):
            if i == 0:
                input_ids = input_ids
            else:
                input_ids = torch.as_tensor([[output_ids[-1]]], device=self.args.device_map)
                # attention_mask = torch.ones(
                #     1, past_key_values[0][0].shape[-2] + 1, device="cuda")
            out = model.forward(input_ids=input_ids,
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        images=image_clip,
                        region_masks=region_masks,
                        valid_region_masks_bool=valid_region_masks_bool,
                        output_hidden_states = True,
                        return_dict=True,
                        )
            logits = out.logits
            past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)
            pred_ids.append(token)

            if stop_idx is not None and token == stop_idx:
                stopped = True
            elif token == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            all_output_hidden_states.append(out.hidden_states[-1])



            if i % args.stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                cur_out = tokenizer.decode(pred_ids, skip_special_tokens=True)
                pos = cur_out.rfind(stop_str)
                if pos != -1:
                    cur_out = cur_out[:pos]
                    stopped = True
                output = ori_prompt + cur_out

                if stopped:
                    ## gen mask if the output is a segment token
                    output_ids_tensor = torch.tensor(output_ids, dtype=torch.long, device=self.args.device_map)
                    seg_token_mask = output_ids_tensor[1:] == self.seg_token_idx
                    seg_token_mask = torch.tensor(seg_token_mask, dtype=torch.bool).cuda().unsqueeze(0)
                    rrres = torch.sum(seg_token_mask)
                    if torch.sum(seg_token_mask) != 0:
                        output_hidden_states = torch.cat(all_output_hidden_states, dim=1)
                        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
                        seg_token_mask = torch.cat(
                            [
                                torch.zeros((seg_token_mask.shape[0], 575)).bool().cuda(),
                                seg_token_mask,
                            ],
                            dim=1,
                        )

                        hidden_states = []

                        assert len(self.model.base_model.text_hidden_fcs) == 1
                        if isinstance(output_hidden_states, tuple):
                            hidden_states.append(self.model.base_model.text_hidden_fcs[0](output_hidden_states[-1]))
                        else:
                            hidden_states.append(self.model.base_model.text_hidden_fcs[0](output_hidden_states))

                        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
                        # B*q_num, N, 256 --> B*q_num, 255+sequence_length, 256
                        pred_embeddings = last_hidden_state[seg_token_mask]

                        # use the first hidden state when there are multiple seg tokens
                        if pred_embeddings.shape[0] > 1:
                            pred_embeddings = pred_embeddings[:1, :]
                        # use the last hidden state when there is no seg tokens
                        elif pred_embeddings.shape[0]:
                            pred_embeddings = last_hidden_state[:1, -2:-1, :].squeeze(1)

                        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

                        image_embeddings = self.model.get_visual_embs(image_sam)

                        multimask_output = False
                        pred_masks = []
                        for i in range(len(pred_embeddings)):
                            # print(pred_embeddings[i].shape)
                            (
                                sparse_embeddings,
                                dense_embeddings,
                            ) = self.model.base_model.visual_model.prompt_encoder(
                                points=None,
                                boxes=None,
                                masks=None,
                                text_embeds=pred_embeddings[i].unsqueeze(0).unsqueeze(1),
                            )

                            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                            low_res_masks, iou_predictions = self.model.base_model.visual_model.mask_decoder(
                                image_embeddings=image_embeddings[i].unsqueeze(0),
                                image_pe=self.model.base_model.visual_model.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=multimask_output,
                            )
                            # pred_mask = self.model.base_model.visual_model.postprocess_masks(
                            pred_mask = self.model.postprocess_masks(
                                low_res_masks,
                                input_size=resize_list[i],
                                original_size=original_size_list[i].shape,
                            )
                            pred_masks.append(pred_mask[:, 0])

                        pred_mask = (torch.sigmoid(pred_masks[0]) > 0.1).int().squeeze(0).cpu().numpy()

                        height, width = pred_mask.shape
                        def encode_sparse(mask):
                            non_zero_positions = np.transpose(np.nonzero(mask))
                            encoded_data = non_zero_positions.tolist()
                            return encoded_data
                        encoded_mask = encode_sparse(pred_mask)
                else:
                    encoded_mask = []
                    height = width = 0

                ret = {
                    "text": output,
                    "mask": encoded_mask,
                    "height": str(height),
                    "width": str(width),
                    "error_code": 0,
                }
                yield json.dumps(ret).encode() + b"\0"

            if stopped:
                break

        if past_key_values is not None:
            del past_key_values
# -------------------------------------------------------------------------------

    def generate_stream_gate(self, params):
        # try:
        for x in self.generate_stream(params):
            yield x
        # except ValueError as e:
        #     print("Caught ValueError:", e)
        #     ret = {
        #         "text": server_error_msg,
        #         "error_code": 1,
        #     }
        #     yield json.dumps(ret).encode() + b"\0"
        # except torch.cuda.CudaError as e:
        #     print("Caught torch.cuda.CudaError:", e)
        #     ret = {
        #         "text": server_error_msg,
        #         "error_code": 1,
        #     }
        #     yield json.dumps(ret).encode() + b"\0"
        # except Exception as e:
        #     print("Caught Unknown Error", e)
        #     ret = {
        #         "text": server_error_msg,
        #         "error_code": 1,
        #     }
        #     yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default='medplib')
    parser.add_argument("--vision_pretrained", type=str, default='../huggingface_models/sam-med2d_b.pth')
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `model.medplib` is included in the model path.")
    parser.add_argument("--keep-aspect-ratio", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-fp16", action="store_true")
    parser.add_argument("--add_region_feature", action="store_true")
    parser.add_argument("--image_w", type=int, default=336)  #  224
    parser.add_argument("--image_h", type=int, default=336)  #  224
    parser.add_argument("--device_map", type=str, default='cpu')
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `model.medplib` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.load_fp16,
                         args.keep_aspect_ratio,
                         args.num_gpus,
                         args.add_region_feature,
                         args.image_w,
                         args.image_h,
                         args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
