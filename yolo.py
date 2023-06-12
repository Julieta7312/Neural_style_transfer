import torch
import numpy as np
import cv2
import os, time
import pandas as pd
from PIL import Image



IMAGES_PATH = './to_yolo/dataset/images'
IMAGES_PRED_PATH = './to_yolo/dataset/preds'
PATH_LAST_PT = './fine_tuned_mdl/weights/last.pt'
PARAMS = dict(repo_or_dir='ultralytics/yolov5', model='custom', path=PATH_LAST_PT, force_reload=True)

fine_mdl = torch.hub.load(**PARAMS)

xyxys = []
file_names = list( map(lambda x: x.split('.jpg')[0] if x.endswith('.jpg') else '0', os.listdir(IMAGES_PATH)) )
for i in file_names:
    img_path = f"{IMAGES_PATH}/{i}.jpg"
    if os.path.exists(img_path):
        res = fine_mdl(f"{IMAGES_PATH}/{i}.jpg")
        im = Image.fromarray(np.squeeze(res.render()))
        im.save(f"{IMAGES_PRED_PATH}/{i}.jpg")
        res_df = res.pandas().xyxy[0]
        res_df.index = [i]*res_df.shape[0]
        xyxys.append(res_df)
boxes_classes = pd.concat(xyxys, axis=0)
boxes_classes.to_csv(f"{IMAGES_PRED_PATH}/yolo_total_res.csv")



