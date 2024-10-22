from glob import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

def test(args):
    # 1. 모델 불러오기 (필요한 모델 개수에 맞춰 추가 가능)
    model_paths = glob(f"{args.model_dir}/*.pt")
    models = [YOLO(path) for path in model_paths]

    # 2. 테스트 데이터셋 경로 설정
    test_data = args.test_data

    # 3. 저장 경로 설정 및 폴더 생성
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 4. 각 모델에 대해 성능 평가
    metrics = []
    for model in models:
        result = model.val(data=test_data)
        metrics.append({
            'model': model.model,
            'map50': result['map50'],
            'precision': result['precision'],
            'recall': result['recall']
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
    models_names = [f"Model {i+1}" for i in range(len(models))]
    map50 = [metric['map50'] for metric in metrics]
    precision = [metric['precision'] for metric in metrics]
    recall = [metric['recall'] for metric in metrics]

    # mAP 시각화 및 저장
    plt.bar(models_names, map50, color='blue')
    plt.title('mAP@0.5 Comparison')
    plt.xlabel('Models')
    plt.ylabel('mAP@0.5')
    map50_plot_path = os.path.join(save_dir, "map50_comparison.png")
    plt.savefig(map50_plot_path)
    plt.close()  # 그래프를 닫아서 메모리 해제

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
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save plots and metrics")
    args = parser.parse_args()
    
    test(args)