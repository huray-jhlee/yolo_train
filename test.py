from ultralytics import YOLO
import matplotlib.pyplot as plt

# 1. 모델 불러오기 (필요한 모델 개수에 맞춰 추가 가능)
model_paths = ["model1.pt", "model2.pt", "model3.pt"]
models = [YOLO(path) for path in model_paths]

# 2. 테스트 데이터셋 경로 설정
test_data = "test_dataset.yaml"

# 3. 각 모델에 대해 성능 평가
metrics = []
for model in models:
    result = model.val(data=test_data)
    metrics.append({
        'map50': result['map50'],
        'precision': result['precision'],
        'recall': result['recall']
    })

# 4. 모델별 성능 지표 출력
for i, metric in enumerate(metrics):
    print(f"Model {i+1} Performance:")
    print(f"  mAP@0.5: {metric['map50']}")
    print(f"  Precision: {metric['precision']}")
    print(f"  Recall: {metric['recall']}")
    print()

# 5. 성능 지표 시각화
models_names = [f"Model {i+1}" for i in range(len(models))]
map50 = [metric['map50'] for metric in metrics]
precision = [metric['precision'] for metric in metrics]
recall = [metric['recall'] for metric in metrics]

# mAP 시각화
plt.bar(models_names, map50, color='blue')
plt.title('mAP@0.5 Comparison')
plt.xlabel('Models')
plt.ylabel('mAP@0.5')
plt.show()

# Precision 시각화
plt.bar(models_names, precision, color='green')
plt.title('Precision Comparison')
plt.xlabel('Models')
plt.ylabel('Precision')
plt.show()

# Recall 시각화
plt.bar(models_names, recall, color='red')
plt.title('Recall Comparison')
plt.xlabel('Models')
plt.ylabel('Recall')
plt.show()