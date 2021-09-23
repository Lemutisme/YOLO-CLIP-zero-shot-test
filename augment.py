import numpy as np
import pandas as pd
from PIL import Image
import torch
from model import get_clip
import cv2
import requests
import os
from yolo_crop import process

FRAME_SKIP = 5

def find_best_matches(text_features, photo_features, photo_ids, results_count, device):
  similarities = (photo_features @ text_features.T).squeeze(1)
  best_photo_idx = (-similarities).argsort()
  return [photo_ids[i] for i in best_photo_idx[:results_count]]

def process_video(video_path, save_path):
  # 载入视频，yolo处理帧
  print('Processing video ...')
  vd = cv2.VideoCapture(video_path)
  try:
    os.mkdir('video_frames')
  except:
    print('Directory already exsists')
  count = 0
  while True:
    # read the next frame from the file
    (grabbed, frame) = vd.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
      break
    count += 1
    if count % FRAME_SKIP == 0:
      cv2.imwrite(os.path.join('video_frames', f'{count}.jpg'), frame)
  process('video_frames', save_path)
  
    


def image_to_image(video_path):
  process_video(video_path, 'augment_crops')

  # clip，unsplash图片特征及id
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = get_clip(device)
  photo_ids = pd.read_csv("unsplash-dataset/photo_ids.csv")
  photo_ids = list(photo_ids['photo_id'])
  photo_features = np.load("unsplash-dataset/features.npy")
  photo_features = torch.tensor(photo_features).to(device)
  try:
    os.mkdir('aug_images')
  except:
    print('Directory already exsists')

  # 收集yolo识别过的图片
  jpgs = []
  for root, dirs, files in os.walk('crops'):
    for f in files:
      if f.endswith('.jpg'):
        jpgs.append(os.path.join(root, f))

  ids = []
  print('Searching ...')
  count = 0
  for j in jpgs:
    with torch.no_grad():
      image_feature = model.encode_image(preprocess(Image.open(j)).unsqueeze(0).to(device))
      image_feature = (image_feature / image_feature.norm(dim=-1, keepdim=True))
    best_photo_ids = find_best_matches(image_feature, photo_features, photo_ids, 1, device)
    ids += best_photo_ids
    count += 1
    print(f'Image {count} searched')

  print('Downloading ...')
  # 下载相似图片
  for photo_id in ids:
    response = requests.get(f'https://unsplash.com/photos/{photo_id}/download')
    with open(f'aug_images/{photo_id}.jpg', 'wb') as f:
      f.write(response.content)
    print(f'{photo_id} downloaded')


if __name__ == '__main__':
  import sys
  image_to_image(sys.argv[1])