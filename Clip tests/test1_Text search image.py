# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:13:18 2021

@author: Clause
"""
import torch
import clip
from PIL import Image


device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image1 = preprocess(Image.open(r"D:\****dmx2m6lfgx.jpg")).unsqueeze(0)
image2 = preprocess(Image.open(r"D:\****hyqla0ug9.jpg")).unsqueeze(0)
image3 = preprocess(Image.open(r"D:\****uo39v1va.jpg")).unsqueeze(0)
text = clip.tokenize(["football"]).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
image = torch.cat([image1, image2, image3]).to(device)
print(image)
print(text)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    print(logits_per_image, logits_per_text)
    probs = logits_per_text.softmax(dim=-1).cpu().numpy()
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
