
import os
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

from ultralytics import YOLO

def plot_combined_metrics(df, save_dir):
    """
    각 모델의 결과를 하나의 그래프에 플롯.
    """
    metrics = ["mp", "mr", "map50", "map5095"]
    
    for metric in metrics:
        plt.figure()
        for model_name in df["model"].unique():
            model_data = df[df["model"] == model_name]
            plt.plot(model_data["confidence"], model_data[metric], marker="o", label=model_name)
            
            # 각 점 위에 값 표시
            for x, y in zip(model_data["confidence"], model_data[metric]):
                plt.text(x, y, f"{y:.3f}", fontsize=8, ha="center", va="bottom")
        
        plt.xlabel("Confidence Threshold")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Confidence Threshold (All Models)")
        plt.grid(True)
        plt.legend()
        
        # Plot 저장
        plot_path = os.path.join(save_dir, f"{metric}_vs_confidence_combined.png")
        plt.savefig(plot_path)
        print(f"Combined plot saved to {plot_path}")
        plt.close()

def optimizing_model(model_dir, device_id, test_yaml_path, save_dir):
    
    device = f"cuda:{device_id}" if device_id is not None and torch.cuda.is_available() else "cpu"
    
    model_paths = glob(f"{model_dir}/*.pt")
    model_paths = sorted(model_paths)
    model_dict = {os.path.basename(path).replace(".pt", ""):YOLO(path).to(device) for path in model_paths}
    
    # model = YOLO(model_path).to(device)
    
    results = []
    metric_names = ["mp", "mr", "map50", "map5095"]
    conf_thresholds = [0.001, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    
    for model_name, model in model_dict.items():
        for conf_threshold in conf_thresholds:
            result = model.val(
                data=test_yaml_path,
                conf = conf_threshold,
                iou=0.6,
                project=os.path.join(save_dir, model_name),
                name=f"{conf_threshold}"
            )
            ####
            result_dict = {name: value for name, value in zip(metric_names, result.mean_results())}
            result_dict["confidence"] = conf_threshold  # Confidence Threshold 추가
            result_dict["model"] = model_name
            results.append(result_dict)
            
    df = pd.DataFrame(results)
    
    csv_path = os.path.join(save_dir, "confidence_optimization_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Plot 생성
    plot_combined_metrics(df, save_dir)
    
    
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/ai04/jh/codes/yolo_train/test/0_opt_test/0_models")
    parser.add_argument("--device", type=str, default=4)
    parser.add_argument("--data", type=str, default="/data2/jh/241019/test/test.yaml")
    parser.add_argument("--save_dir", type=str, default="/home/ai04/jh/codes/yolo_train/test/0_opt_test")
    
    args = parser.parse_args()
    
    optimizing_model(args.model, args.device, args.data, args.save_dir)