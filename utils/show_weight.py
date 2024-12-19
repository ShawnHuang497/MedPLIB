import torch

weight_path = "/root/huggingface_models/llava-v1.5-13b-pretrain-bs128/mm_projector.bin"
weight_path = "/root/huggingface_models/sam-med2d_b.pth"
weight_path = "/root/huggingface_models/clip-vit-large-patch14-336/pytorch_model.bin"
weight_path = "/root/paddlejob/workspace/env_run/output/LISA/runs/lisa-13b-bird-0417/pytorch_model.bin"
state_dict = torch.load(weight_path, map_location=torch.device('cpu'))  # 加载权重文件
# 打印模型参数
print(weight_path)
print(state_dict.keys())
# print(state_dict['model'].keys())
# for name, param in state_dict['model'].items():
for name, param in state_dict.items():
    print(name, param.shape)
    print(param)