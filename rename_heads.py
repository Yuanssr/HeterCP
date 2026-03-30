#!/usr/bin/env python3
import argparse
import torch

def rename_keys(state):
    renamed = {}
    for k, v in state.items():
        if k.startswith("reg_head."):
            renamed["reg_head_m1." + k.split(".", 1)[1]] = v
        elif k.startswith("cls_head."):
            renamed["cls_head_m1." + k.split(".", 1)[1]] = v
        elif k.startswith("dir_head."):
            renamed["dir_head_m1." + k.split(".", 1)[1]] = v
        else:
            renamed[k] = v
    return renamed

def main(in_path, out_path):
    ckpt = torch.load(in_path, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt["state_dict"] = rename_keys(ckpt["state_dict"])
    else:
        ckpt = rename_keys(ckpt)
    torch.save(ckpt, out_path)
    print(f"saved to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    args = ap.parse_args()
    main(args.in_path, args.out_path)