import os
import cv2
import yaml
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import torch
from ultralytics import YOLO

os.environ["OMP_NUM_THREADS"] = '1'

def draw_bbox(image, bboxes):
    """
    이미지를 불러와서 주어진 바운딩 박스를 그린 후 저장하는 함수
    
    Parameters:
    - image_path: 이미지 파일 경로
    - bboxes: 바운딩 박스 리스트, 각 바운딩 박스는 [x_min, y_min, x_max, y_max] 형식
    - output_path: 바운딩 박스가 그려진 이미지를 저장할 경로
    """
    # 이미지 불러오기
    # image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 바운딩 박스 그리기
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        # 바운딩 박스를 그린다. 색상은 초록색(0, 255, 0), 두께는 2
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 10)
    
    return image

device_num = 3
device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"

model_paths = glob("/home/ai04/jh/codes/yolo_train/test/models/*.pt")
model_dict = {os.path.basename(path):YOLO(path).to(device) for path in model_paths}

test_data = '/data2/jh/241019/test/test.yaml'
with open(test_data, "r") as f:
    data = yaml.safe_load(f)

image_dir = data['test']
img_paths = glob(os.path.join(image_dir, "*.jpg"))

result_list = []

sampled_img_paths = np.random.choice(img_paths, 10000, replace=False)

for img_path in tqdm(sampled_img_paths):

    # food_lens
    food_lens_label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
    
    with open(food_lens_label_path, "r") as f:
        food_lens_labels = f.readlines()
    
    food_lens_labels = [[float(x_) for x_ in x.strip().split(" ")] for x in food_lens_labels]
    
    inner_dict = {
        "food_lense" : food_lens_labels
    }
    # pred yolo

    for model_name, model in model_dict.items():
        
        results = model(img_path, verbose=False)
        if "image" not in inner_dict:
            inner_dict["image"] = results[0].orig_img
            
        inner_dict[model_name] = list(zip(results[0].boxes.cls, results[0].boxes.xyxy))
        
    result_list.append(inner_dict)


with open("results_list.pkl", "wb") as f:
    pickle.dump(result_list, f, pickle.HIGHEST_PROTOCOL)