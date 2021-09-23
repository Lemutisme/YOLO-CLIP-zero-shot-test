#把相关图片或文本 encode进行保存
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 加载数据
kkk_dict_all = np.load(r"D:\t***1.npy", allow_pickle=True).item()

# 加载模型
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 计算缩略图向量
aidss=[]
titles=[]
for i, j in kkk_dict_all.items():

    try:

        aidss.append(i)
        title1 = j[0]
        print(title1)
        text1 = clip.tokenize([title1]).to(device)
        print(text1.shape)
        with torch.no_grad():
            text_features = model.encode_text(text1)
            # text_features /= text_features.norm(dim=-1, keepdim=True)
            # print(text_features)
            titles.append(text_features)
            print(type(text_features))

    except Exception as e:
        print(e)
        print("####")
        aidss.pop()
        pass

# 保存
np.save(r"D:\t**title_aidss.npy", aidss)
np.save(r"D:\***\title_features_embs.npy", titles)
