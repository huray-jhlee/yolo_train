import os
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from ultralytics import YOLO

def test(args):
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    
    # 1. 모델 불러오기 (필요한 모델 개수에 맞춰 추가 가능)
    model_paths = glob(f"{args.model_dir}/*.pt")
    model_paths = sorted(model_paths)
    models = {os.path.basename(path):YOLO(path).to(device) for path in model_paths}

    # 2. 테스트 데이터셋 경로 설정
    test_data = args.test_data

    # 3. 저장 경로 설정 및 폴더 생성
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 4. 각 모델에 대해 성능 평가
    metrics = []
    for model_name, model in models.items():
        result = model.val(
            data=test_data,
            project=save_dir,
            name=model_name,
        )
        mp, mr, map50, map = result.mean_results()
        
        metrics.append({
            'model': model_name,
            'map50': map50,
            'map': map,
            'precision': mp,
            'recall': mr,
        })
    # 5. 성능 지표 출력 및 저장
    metrics_df = pd.DataFrame(metrics)
    metrics_csv_path = os.path.join(save_dir, "metrics.csv")
    metrics_pkl_path = os.path.join(save_dir, "metrics.pkl")

    metrics_df.to_csv(metrics_csv_path, index=False)  # CSV로 저장
    with open(metrics_pkl_path, 'wb') as pkl_file:
        pickle.dump(metrics, pkl_file)  # PKL로 저장

    print(f"Performance metrics saved to {metrics_csv_path} and {metrics_pkl_path}")

    # 6. 성능 지표 시각화 및 저장
    # models_names = [f"Model {i+1}" for i in range(len(models))]
    models_names = list(models.keys())
    map50 = [metric['map50'] for metric in metrics]
    map_all = [metric['map'] for metric in metrics]  # mAP@0.5:0.95
    precision = [metric['precision'] for metric in metrics]
    recall = [metric['recall'] for metric in metrics]

    # mAP@0.5 시각화 및 저장
    plt.bar(models_names, map50, color='blue')
    plt.title('mAP@0.5 Comparison')
    plt.xlabel('Models')
    plt.ylabel('mAP@0.5')
    map50_plot_path = os.path.join(save_dir, "map50_comparison.png")
    plt.savefig(map50_plot_path)
    plt.close()

    # mAP@0.5:0.95 시각화 및 저장
    plt.bar(models_names, map_all, color='cyan')
    plt.title('mAP@0.5:0.95 Comparison')
    plt.xlabel('Models')
    plt.ylabel('mAP@0.5:0.95')
    map_plot_path = os.path.join(save_dir, "map_comparison.png")
    plt.savefig(map_plot_path)
    plt.close()

    # Precision 시각화 및 저장
    plt.bar(models_names, precision, color='green')
    plt.title('Precision Comparison')
    plt.xlabel('Models')
    plt.ylabel('Precision')
    precision_plot_path = os.path.join(save_dir, "precision_comparison.png")
    plt.savefig(precision_plot_path)
    plt.close()

    # Recall 시각화 및 저장
    plt.bar(models_names, recall, color='red')
    plt.title('Recall Comparison')
    plt.xlabel('Models')
    plt.ylabel('Recall')
    recall_plot_path = os.path.join(save_dir, "recall_comparison.png")
    plt.savefig(recall_plot_path)
    plt.close()

    print(f"Plots saved to {save_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_data", type=str, default="/data2/jh/241019/test/test.yaml")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save plots and metrics")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    
    test(args)