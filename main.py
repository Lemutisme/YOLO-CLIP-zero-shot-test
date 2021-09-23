from yolo_crop import process
import sys

def yolo_clip_predict():
  process(sys.argv[1], 'crops')

  gt, pred = predict('crops')

  count = 0
  for i in range(len(gt)):
      if gt[i] == pred[i]:
          count += 1

  print(f'Accuracy: {count / len(gt)}')