import os
import torch
from model import get_yolo

def collect_jpg(path):
    """遍历目录，收集图片"""
    jpgs = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.jpg'):
                jpgs.append(os.path.join(root, f))     
    return jpgs

def process(path, save_path):
    """yolo剪裁图片并保存"""
    jpgs = collect_jpg(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_yolo(device)
    for j in jpgs:
        results = model(j)
        results.crop(save_path)