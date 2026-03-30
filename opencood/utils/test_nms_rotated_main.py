import time
import numpy as np
import torch
from opencood.utils import box_utils

def gen_rotated_boxes(n=8000, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    cx = rng.uniform(-50, 50, size=n)
    cy = rng.uniform(-50, 50, size=n)
    w = rng.uniform(1.5, 4.0, size=n)
    h = rng.uniform(1.5, 4.0, size=n)
    ang = rng.uniform(-np.pi, np.pi, size=n)
    boxes = []
    for i in range(n):
        c, s = np.cos(ang[i]), np.sin(ang[i])
        pts = np.array([[ w[i]/2,  h[i]/2],
                        [ w[i]/2, -h[i]/2],
                        [-w[i]/2, -h[i]/2],
                        [-w[i]/2,  h[i]/2]])
        R = np.array([[c, -s],[s, c]])
        pts = (R @ pts.T).T + np.array([cx[i], cy[i]])
        boxes.append(pts)
    boxes = torch.tensor(np.stack(boxes, axis=0), dtype=torch.float32, device=device)  # (N,4,2)
    scores = torch.rand(n, device=device)
    return boxes, scores

def bench_once(fn, boxes, scores, name):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    keep = fn(boxes, scores)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"{name}: kept {len(keep)} / {boxes.shape[0]}, time {t1-t0:.4f}s")
    return keep

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    boxes, scores = gen_rotated_boxes(n=8000, seed=1, device=device)
    thr = 0.1

    print(f"device={device}")
    # CPU 版（一定可用）
    bench_once(lambda b, s: box_utils.nms_rotated(b, s, thr), boxes, scores, "nms_rotated (CPU)")

    # MMCV 版（若可用走 GPU，否则自动回退 CPU）
    bench_once(lambda b, s: box_utils.nms_rotated_mmcv(b, s, thr, top_k=8000), boxes, scores, "nms_rotated_mmcv (GPU/CPU)")

if __name__ == "__main__":
    main()