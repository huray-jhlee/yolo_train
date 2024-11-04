import os
import cv2
import time
import torch
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO

def data_load(data_dir, sample_num):
    if sample_num == 100:
        with open(os.path.join(data_dir, "sample_100.txt"), "r") as f:
            sampled_datas = f.readlines()
    elif sample_num == 1000:
        with open(os.path.join(data_dir, "sample_1000.txt"), "r") as f:
            sampled_datas = f.readlines()
    
    sampled_datas = [x.strip() for x in sampled_datas]
    
    return sampled_datas

def model_load(model_dir, device=None):
    
    if device is None:
        device = "cpu"
    elif torch.cuda.is_available() :
        device = f"cuda:{device}"
    else :
        device = "cpu"
    
    model_paths = glob(os.path.join(model_dir, "**"), recursive=True)
    model_paths = [x for x in model_paths if os.path.isfile(x)]
    
    model_dict = {}
    for model_path in model_paths:
        filename = os.path.basename(model_path).split(".")[0]
        model_dict[filename] = YOLO(model_path).to(device)
    
    return model_dict

def main(args):
    result_metric = ["mp", "mr", "map50", "map"]
    model_dict = model_load(args.model_dir, args.device)
    sampled_datas = data_load(args.data_dir, args.sample_num)
    print(model_dict.keys())
    
    print("Calculate metrics..")
    # calc metric
    for model_name, model in model_dict.items():
        val_results = model.val(
            data=args.yaml,
            batch=100,
            project=args.save_dir,
            name=f"val-{model_name}"
        )
        
        time.sleep(0.5)
        
        result_list = [f"{metric}: {value}" for metric, value in zip(result_metric, val_results.mean_results())]
        result_save_dir = os.path.join(args.save_dir, f"val-{model_name}")
        os.makedirs(result_save_dir, exist_ok=True)
        
        with open(os.path.join(result_save_dir, "results.txt"), "w") as f:
            f.write("\n".join(result_list))
    
    print("Start Inference and save imgs..")
    # inference + plot
    for model_name, model in model_dict.items():
        pred_results = model.predict(
            sampled_datas
        )
        
        img_save_dir = os.path.join(args.save_dir, f"val-{model_name}", "images")
        os.makedirs(img_save_dir, exist_ok=True)
        
        for i, r in tqdm(enumerate(pred_results), total=len(pred_results)):
            im_bgr = r.plot()
            cv2.imwrite(os.path.join(img_save_dir, f"results{i}.jpg"), im_bgr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/ai04/jh/codes/yolo_train/test/tmp_codes")
    parser.add_argument("--model_dir", type=str, default="/home/ai04/jh/codes/yolo_train/test/exp2_models")
    parser.add_argument("--yaml", type=str, default="/home/ai04/jh/codes/yolo_train/test/tmp_codes/sampled_100.yaml")
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--sample-num", type=int, default=100, help="100 or 1000")
    parser.add_argument("--save_dir", type=str, default="./test/sample_inference_best")
    args = parser.parse_args()
    
    main(args)