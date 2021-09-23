import torch
import clip

def get_yolo(device):
    """载入yolo模型, 支持cuda时用cuda"""
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True) 
    model.eval()
    model.to(device)
    return model

def get_clip(device):
    """载入clip模型"""
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess