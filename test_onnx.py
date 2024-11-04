import os
import cv2
import time
import numpy as np
from glob import glob

from ultralytics import YOLO

from utils import load_yaml_to_dict, img_preprocess, get_gt_info, process_model_metrics, \
    calculate_map50_and_map5090, compute_precision_recall, compute_tp_fp_fn, draw_bboxes_on_image

os.environ["ORT_LOG_LEVEL"] = "DEBUG"

def test(args):
    
    model_paths = glob(f"{args.model_dir}/*.pt")
    models = {os.path.basename(path): YOLO(path).to("cpu")for path in model_paths}

    onnx_model_paths = glob(f"{args.model_dir}/*.onnx")
    onnx_models = {os.path.basename(path): YOLO(path) for path in onnx_model_paths}
    
    models.update(onnx_models)
    
    # 2. 테스트 데이터셋 경로 설정
    test_data = args.test_data
    
    yaml_dict = load_yaml_to_dict(test_data)
    test_dir = yaml_dict["test"]
    test_img_paths = glob(os.path.join(test_dir, "*.jpg"))

    # 3. 저장 경로 설정 및 폴더 생성
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 4. 각 모델에 대해 성능 평가
    sampled_test_img_paths = np.random.choice(test_img_paths, args.sample_num, replace=False)
    img_batches = [sampled_test_img_paths[i:i+args.batch] for i in range(0, len(sampled_test_img_paths), args.batch)]

    model_metrics = {
        model_name: {
            "precision": [],
            "recall": [],
            "map50": [],
            "map5090": [],
            "inference_time":[]
        } for model_name in list(models.keys())
    }
    
    for img_batch in img_batches:
        batch_gt_class_ids, batch_gt_boxes, img_size_batch = get_gt_info(img_batch)
        batch_img_tensor = img_preprocess(img_batch)
        
        for model_name, model in models.items():
            
            start_time = time.time()
            batch_pred = model.predict(batch_img_tensor) if ".onnx" not in model_name else model.predict(batch_img_tensor, device="cpu")
            inf_time = time.time() - start_time
            print("=="*30)
            print(model_name)
            print(inf_time)
            print("=="*30)
             
            batch_pred_boxes = [pred.boxes.xyxy.numpy() for pred in batch_pred]
            batch_pred_class_ids = [pred.boxes.cls.numpy() for pred in batch_pred]
            batch_pred_scores = [pred.boxes.conf.numpy() for pred in batch_pred]
            
            # debug last batch, first image
            tmp_img = batch_img_tensor[0].permute(1, 2, 0).numpy()
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            tmp_bboxes = batch_pred_boxes[0]
            c_img = draw_bboxes_on_image(tmp_img, tmp_bboxes)
            cv2.imwrite(os.path.join(args.save_dir, f"{model_name}.png"), c_img)
                
                
            tp, fp, fn = compute_tp_fp_fn(batch_pred_boxes, batch_pred_class_ids, batch_pred_scores,
                                            batch_gt_boxes, batch_gt_class_ids)
            
            precision, recall = compute_precision_recall(tp, fp, fn)
            map50, map5090 = calculate_map50_and_map5090(batch_pred_boxes, batch_pred_class_ids, batch_pred_scores, batch_gt_boxes, batch_gt_class_ids)

            model_metrics[model_name]["precision"].append(precision)
            model_metrics[model_name]["recall"].append(recall)
            model_metrics[model_name]["map50"].append(map50)
            model_metrics[model_name]["map5090"].append(map5090)
            model_metrics[model_name]["inference_time"].append(inf_time)
    
    process_model_metrics(model_metrics, os.path.join(args.save_dir, "results.csv"), args.save_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--test_data", type=str, default="/data2/jh/241019/test/test.yaml")
    parser.add_argument("--save_dir", type=str, default="./test/test99/", help="Directory to save plots and metrics")
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    
    test(args)