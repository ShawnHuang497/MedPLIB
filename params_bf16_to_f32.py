import torch
import argparse
import os

def load_model_parameters(directory, device):
    combined_params = {}

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith("model_states.pt"):
            filepath = os.path.join(directory, filename)
            
            # 检查文件名是否包含"expert"
            if "expert" in filename:
                # 直接加载模型参数
                state_dict = torch.load(filepath, map_location=device)
            else:
                # 模型参数在"module"键下面
                loaded_data = torch.load(filepath, map_location=device)
                state_dict = loaded_data['module']
            
            # 合并模型参数到一个字典中
            for k, v in state_dict.items():
                if k in combined_params:
                    raise ValueError(f"Duplicate key found in state dicts: {k}")
                combined_params[k] = v.to(torch.float32)  # 转换为float32

    return combined_params

def save_model_parameters(parameters, save_path):
    torch.save(parameters, save_path)

def main():
    parser = argparse.ArgumentParser(description="Merge and convert model parameters.")
    parser.add_argument('--directory', type=str, required=True, help='Directory containing the model state files.')
    args = parser.parse_args()

    # 设定保存路径为输入目录的上一级目录
    save_path = os.path.join(os.path.dirname(os.path.dirname(args.directory)), 'pytorch_model.bin')

    # 加载并合并模型参数
    combined_parameters = load_model_parameters(args.directory, 'cpu')

    # 保存合并后的模型参数
    save_model_parameters(combined_parameters, save_path)

    print(f"Combined model parameters saved to {save_path}")

if __name__ == "__main__":
    main()
