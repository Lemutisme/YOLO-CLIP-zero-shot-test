import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#
# for i in tqdm(range(1000),20):
#     print(i)
kkk_all =  np.load(r"D:****图\媒资内容1.npy")
aids = [ i[0] for i in kkk_all.tolist()]
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
aidss=[]
imgs=[]
for i in tqdm(aids):
    try:
        # print(i)
        aidss.append(i)
        image1 = preprocess(Image.open(r"D:***图\图\{}.jpg".format(i))).unsqueeze(0)
        with torch.no_grad():
            image_features = model.encode_image(image1)
            imgs.append(image_features.numpy())
            print(type(image_features))
        # imgs.append(image1)
    except Exception as e:
        print(e)
        print("33333####")
        aidss.pop()
        pass


np.save(r"D:\***图\aidss.npy", aidss)
np.save(r"D:***图\image_features_embs.npy", imgs)
