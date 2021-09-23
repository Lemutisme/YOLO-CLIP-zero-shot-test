#加载测试
import torch
import clip
from PIL import Image

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

image1 = np.load(r"D:\***图\image_features_embs.npy", allow_pickle=True)
aidss = np.load(r"D:\***图\aidss.npy")
print(aidss)

device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


text = clip.tokenize(["football"]).to(device)

print(text)
image = torch.Tensor([(item / item.norm(dim=-1, keepdim=True)).cpu().numpy() for item in image1])
print(image)
print(type(image))


with torch.no_grad():
    # image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # text_features = text_features.cpu().numpy()
    # print(text_features)
    # print(image_features.shape)
    # print(image_features)
    # print(text_features.shape)
    similarities = (image @ text_features.T).squeeze(1)
    print(similarities[:,0])
    best_photo_idx = np.argsort(similarities[:, 0].numpy())[::-1]

    print(best_photo_idx)
    print([aidss[i] for i in best_photo_idx[:10]])

