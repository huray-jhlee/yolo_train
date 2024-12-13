import os
import torch
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict

def main(args):
    
    model_paths = glob(os.path.join(args.root_dir, "models", "*.pt"))
    
    with open(args.data, "r") as f:
        img_paths = f.readlines()
    img_paths = [x.strip() for x in img_paths]
    
    print(len(img_paths))
    inference_times = defaultdict(list)
    for model_path in model_paths:
        model_name = os.path.basename(model_path).split(".")[0]
        print(model_name)
        model = YOLO(model_path).to(f"cuda:{args.gpu}")
        for img_path in tqdm(img_paths):
            results = model.predict(img_path, conf=0.25, iou=0.5, verbose=False)
            inference_times[model_name].append(results[0].speed['inference'])
    
    df = pd.DataFrame(inference_times)
    df.to_csv(os.path.join(args.root_dir, "inf_time.csv"))
    df.describe().to_csv(os.path.join(args.root_dir, "inf_time_summary.csv"))
    print(df.describe())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--data", type=str, default="/home/ai04/jh/codes/yolo_train/test/tmp_codes/sample_1000.txt")
    args = parser.parse_args()
    
    main(args)