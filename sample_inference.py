import os
import cv2
import time
import torch
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO

def data_load(data_path: str) -> list:
    
    with open(data_path, "r") as f:
        data_list = f.readlines()
    data_list = [x.strip() for x in data_list]
    
    return data_list

def model_load(model_dir:str, device=None) -> dict:
    
    if device is None:
        device = "cpu"
    elif torch.cuda.is_available() :
        device = f"cuda:{device}"
    else :
        device = "cpu"
    
    model_paths = glob(os.path.join(model_dir, "**"), recursive=True)
    model_paths = [x for x in model_paths if os.path.isfile(x)]
    model_paths = [x for x in model_paths if ".pt" in x.split("/")[-1]]
    model_paths = sorted(model_paths)
    
    model_dict = {}
    for model_path in model_paths:
        filename = os.path.basename(model_path).split(".")[0]
        model_dict[filename] = YOLO(model_path).to(device)
    
    return model_dict

def calc_metric_with_val(model_dict: dict, yaml_path: str, save_dir:str):
    
    print("Calculate metrics..")
    
    result_metric = ["mp", "mr", "map50", "map"]
    
    for model_name, model in model_dict.items():
        val_results = model.val(
            data=yaml_path,
            batch=100,
            project=save_dir,
            name=f"val-{model_name}"
        )
        
        time.sleep(0.5)
        
        result_list = [f"{metric}: {value}" for metric, value in zip(result_metric, val_results.mean_results())]
        save_path = os.path.join(save_dir, f"val-{model_name}", "results.txt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            f.write("\n".join(result_list))
    
    print(f"saved in {os.path.dirname(save_path)}")

def inference_with_sampled_img(model_dict: dict, sampled_datas: list, save_dir: str):
    
    print("Start Inference and save imgs..")
    
    for model_name, model in model_dict.items():
        pred_results = model.predict(
            sampled_datas,
            conf=0.25,
            iou=0.5
            )
    
        img_save_dir = os.path.join(save_dir, f"val-{model_name}", "images")
        os.makedirs(img_save_dir, exist_ok=True)
        
        for i, r in tqdm(enumerate(pred_results), total=len(pred_results)):
            im_bgr = r.plot()
            cv2.imwrite(os.path.join(img_save_dir, f"results{i}.jpg"), im_bgr)
    
    print(f"Inference Done")

def main(args):
    
    model_dict = model_load(args.model_dir, args.device)
    sampled_datas = data_load(args.data_path)
    
    if not args.only_inference:
        calc_metric_with_val(
            model_dict=model_dict,
            yaml_path=args.yaml,
            save_dir=args.save_dir
        )
    
    inference_with_sampled_img(
        model_dict=model_dict,
        sampled_datas=sampled_datas,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/ai04/jh/codes/yolo_train/test/tmp_codes/sample_100.txt")
    parser.add_argument("--model_dir", type=str, default="/home/ai04/jh/codes/yolo_train/test/exp2_models")
    parser.add_argument("--yaml", type=str, default="/home/ai04/jh/codes/yolo_train/test/tmp_codes/sampled_100.yaml")
    parser.add_argument("--device", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./test/sample_inference_best")
    parser.add_argument("--only_inference", action="store_true")
    args = parser.parse_args()
    
    main(args)