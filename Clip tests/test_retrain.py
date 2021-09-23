#https://github.com/openai/CLIP/issues/83

from torch.utils.data import Dataset, DataLoader
import torch
import clip
from torch import nn, optim
import pandas as pd
from PIL import Image

BATCH_SIZE = 5
EPOCH = 50

class image_caption_dataset(Dataset):
    def __init__(self, df):
        self.images = df["image"].tolist()
        self.caption = df["caption"].tolist()
        print(self.images,self.caption)

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        images = preprocess(Image.open(self.images[idx]))
        caption = self.caption[idx]
        return images, caption

df = pd.DataFrame({"image":[r"D:\openai\imgs\gyro warrior.jpg",r"D:\openai\imgs\magic gyro 2.jpg",r"D:\openai\imgs\magic gyro.jpg",r"D:\openai\imgs\most powerful magic gyro.jpg",r"D:\openai\imgs\supervariant battle gyro.jpg"], "caption":["gyro warrior","magic gyro 2","magic gyro","most powerful magic gyro","supervariant battle gyro"]})
print(df)
dataset = image_caption_dataset(df)


train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)  # Define your own dataloader



# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


device = "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)  # Params from paper

for epoch in range(EPOCH):
    for batch in train_dataloader:


        list_image, list_txt = batch  # list_images is list of image in numpy array(np.uint8), or list of PIL images

        # images = torch.stack([preprocess(Image.fromarray(img)) for img in list_image],
        #                      dim=0)  # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
        texts = clip.tokenize(list_txt)
        images = list_image

        logits_per_image, logits_per_text = model(images, texts)
        print(logits_per_image, logits_per_text )
        if device == "cpu":
            ground_truth = torch.arange(BATCH_SIZE).long().to(device)
        else:
            ground_truth = torch.arange(BATCH_SIZE).half().to(device)
        print("######3", ground_truth)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        optimizer.zero_grad()
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        print('[%d] loss: %.3f' %
              (epoch + 1, total_loss))


torch.save(model, r"D:\openai\model1.pkl")
