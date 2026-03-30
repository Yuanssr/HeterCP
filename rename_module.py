import torch
import sys
from collections import OrderedDict

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python rename.py <pth文件路径>")
        sys.exit(1)
    pth_path = sys.argv[1]
    save_path = pth_path.replace(".pth", "_renamed.pth")
    state = torch.load(pth_path, map_location='cpu')
    if isinstance(state, dict) and "model" in state:
        state["model"] = remove_module_prefix(state["model"])
        torch.save(state, save_path)
        loaded = torch.load(save_path, map_location='cpu')
        keys = loaded["model"].keys()
    elif isinstance(state, dict):
        new_state = remove_module_prefix(state)
        torch.save(new_state, save_path)
        loaded = torch.load(save_path, map_location='cpu')
        keys = loaded.keys()
    else:
        print("未知的权重文件格式")
        sys.exit(1)
    print("已保存为", save_path)
    print("新权重文件的所有键：")
    for k in loaded["model"].keys() if "model" in loaded else loaded.keys():
        print(k)