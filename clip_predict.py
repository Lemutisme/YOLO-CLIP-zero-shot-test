import torch
import clip
import os
from PIL import Image
from model import get_clip
from torch.util.data import Dataset, Dataloader

class MDataset(Dataset):
    """自定义Dataset类，配合Dataloader，方便预测或训练"""
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
          idx = idx.tolist()
        return self.X[idx], self.Y[idx]

def build_dataset(path):
    """构建Dataloader和标签集"""
    X = []
    Y = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.jpg'):
                X.append(os.path.join(root, f))
                Y.append(f.split('_')[0])
    return Dataloader(MDataset(X, Y), batch_size=1, shuffle=False), list(set(Y))

def predict(path):
    dataloader, tokens = build_dataset(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = get_clip(device)

    # 预测部分直接参考官方的例子进行小修改
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in tokens]).to(device)
    pred = [] # 预测结果
    gt = [] # ground truth
    for data in dataloader:
        img, label = data
        gt.append(label[0])
        # Prepare the inputs
        image_input = preprocess(Image.open(img[0])).unsqueeze(0).to(device)
        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
        # Pick the top k most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        pred.append(tokens[indices[0]]) # 最接近的label

    return gt, pred