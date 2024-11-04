import os
import cv2
import yaml
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_yaml_to_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

def get_gt_info(img_batch):
    
    def _clean_label(label_path):
        with open(label_path, 'r') as f:
            label_data = f.readlines()
        label_data = [x.strip() for x in label_data]
        return label_data
    
    img_size_batch = [cv2.imread(img_path).shape[:2] for img_path in img_batch]
    
    label_batch = [x.replace("/images/", "/labels/").replace(".jpg", ".txt") for x in img_batch]
    label_batch = [_clean_label(label_path) for label_path in label_batch]
    
    batch_class_ids = []
    batch_boxes = []
    
    for img_size, labels in zip(img_size_batch, label_batch):
        inner_boxes = []
        inner_class_ids = []
        for label in labels:
            splited_label = label.split(" ")
            
            class_id = int(splited_label[0])
            bbox = [float(x) for x in splited_label[1:]]
            inner_class_ids.append(class_id)
            inner_boxes.append(convert_yolo_to_xyxy([bbox], img_size)[0])
        batch_class_ids.append(torch.tensor(inner_class_ids, dtype=torch.int))
        batch_boxes.append([torch.tensor(inner_box) for inner_box in inner_boxes])
        
    batch_boxes = [torch.stack(x) for x in batch_boxes]
    
    return batch_class_ids, batch_boxes, img_size_batch

def img_preprocess(img_batch, input_size=(640, 640)):
    # img2tensor
    batch_img_tensors = []
    for img_path in img_batch:
        img_obj = cv2.imread(img_path)
        img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)
        img_obj = cv2.resize(img_obj, input_size)
        
        img_tensor = torch.tensor(img_obj, dtype=torch.float32).permute(2, 0, 1)
        batch_img_tensors.append(img_tensor)
    
    batch_img_tensors = torch.stack(batch_img_tensors)
    return batch_img_tensors

def convert_yolo_to_xyxy(yolo_boxes, original_shape, resize_shape=(640, 640)):
    """
    YOLO 형식의 바운딩 박스를 예측 형식인 (x1, y1, x2, y2)로 변환하고, 리사이즈된 이미지에 맞게 변환하는 함수.
    
    Args:
    - yolo_boxes: YOLO 형식의 바운딩 박스 리스트 (x_center, y_center, width, height)
    - original_shape: 원본 이미지의 (height, width) 튜플
    - resize_shape: 리사이즈된 이미지의 (height, width) 튜플, 기본값은 (640, 640)
    
    Returns:
    - 변환된 바운딩 박스 리스트 (x1, y1, x2, y2) 형식
    """
    original_height, original_width = original_shape
    resize_height, resize_width = resize_shape
    boxes_xyxy = []
    
    for box in yolo_boxes:
        x_center, y_center, width, height = box
        
        # 원본 이미지에 맞게 절대 좌표로 변환
        x_center_abs = x_center * original_width
        y_center_abs = y_center * original_height
        width_abs = width * original_width
        height_abs = height * original_height
        
        # 리사이즈된 이미지에 맞게 스케일 적용
        x_center_resized = x_center_abs * (resize_width / original_width)
        y_center_resized = y_center_abs * (resize_height / original_height)
        width_resized = width_abs * (resize_width / original_width)
        height_resized = height_abs * (resize_height / original_height)
        
        # 좌상단과 우하단 좌표 계산
        x1 = int(x_center_resized - width_resized / 2)
        y1 = int(y_center_resized - height_resized / 2)
        x2 = int(x_center_resized + width_resized / 2)
        y2 = int(y_center_resized + height_resized / 2)
        
        boxes_xyxy.append([x1, y1, x2, y2])
    
    return boxes_xyxy

def compute_tp_fp_fn(batch_pred_boxes, batch_pred_class_ids, batch_pred_scores, batch_gt_boxes, batch_gt_labels, iou_threshold=0.5, score_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    
    def iou(box1, box2):
        """ 두 박스 간의 IoU 계산 (box1은 단일, box2는 다중) """
        xA = np.maximum(box1[0], box2[:, 0])
        yA = np.maximum(box1[1], box2[:, 1])
        xB = np.minimum(box1[2], box2[:, 2])
        yB = np.minimum(box1[3], box2[:, 3])
        
        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        iou = interArea / (box1Area + box2Area - interArea)
        return iou

    # 각 이미지별로 예측 처리
    for pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels in zip(batch_pred_boxes, batch_pred_class_ids, batch_pred_scores, batch_gt_boxes, batch_gt_labels):
        pred_boxes = np.array(pred_boxes)  # 예측된 박스들을 numpy로 변환
        pred_labels = np.array(pred_labels)  # 클래스 ID
        pred_scores = np.array(pred_scores)  # confidence scores
        gt_boxes = np.array(gt_boxes)  # Ground Truth 박스
        gt_labels = np.array(gt_labels)  # Ground Truth 클래스 ID
        
        detected_gt = np.zeros(len(gt_boxes), dtype=bool)  # 현재 이미지의 Ground Truth 검출 여부 기록

        # Confidence score를 기준으로 필터링
        score_mask = pred_scores >= score_threshold
        filtered_pred_boxes = pred_boxes[score_mask]
        filtered_pred_labels = pred_labels[score_mask]

        # 예측된 각 박스에 대해 IoU를 계산하여 TP, FP 판정
        for pred_box, pred_label in zip(filtered_pred_boxes, filtered_pred_labels):
            ious = iou(pred_box, gt_boxes)  # 모든 Ground Truth와 IoU 계산
            
            # IoU와 클래스가 일치하는지 체크
            iou_mask = (ious >= iou_threshold) & (gt_labels == pred_label)
            
            if np.any(iou_mask):  # TP: 일치하는 GT가 있으면
                tp += 1
                detected_gt[np.argmax(ious)] = True  # 해당 Ground Truth는 검출됨
            else:
                fp += 1  # FP: 일치하는 GT가 없으면

        # FN: 매칭되지 않은 Ground Truth
        fn += np.sum(~detected_gt)

    return tp, fp, fn

def compute_precision_recall(tp, fp, fn):
    """TP, FP, FN을 기반으로 Precision과 Recall 계산"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

def compute_average_precision(precision_list, recall_list):
    """Precision-Recall 곡선의 면적을 계산하여 AP(Average Precision) 반환"""
    precision = np.array(precision_list)
    recall = np.array(recall_list)
    
    # Debugging: Precision과 Recall 값 확인
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    
    # Precision-Recall 값을 순서대로 정렬
    sorted_indices = np.argsort(recall)
    sorted_recall = recall[sorted_indices]
    sorted_precision = precision[sorted_indices]
    
    # np.trapz로 면적 계산
    ap = np.trapz(sorted_precision, sorted_recall)
    
    # Debugging: AP 값 확인
    # print(f"AP: {ap}")
    
    return ap

def calculate_map50_and_map5090(batch_pred_boxes, batch_pred_class_ids, batch_pred_scores, batch_gt_boxes, batch_gt_labels, score_thresholds=np.arange(0.0, 1.0, 0.1), iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    all_precisions = []
    all_recalls = []

    for iou_threshold in iou_thresholds:
        precisions = []
        recalls = []

        # print(f"Calculating for IoU threshold: {iou_threshold}")

        # 다양한 score_threshold에 대한 Precision과 Recall을 구함
        for score_threshold in score_thresholds:
            # print(f"Score threshold: {score_threshold}")
            
            tp, fp, fn = compute_tp_fp_fn(batch_pred_boxes, batch_pred_class_ids, batch_pred_scores, batch_gt_boxes, batch_gt_labels, iou_threshold, score_threshold)
            
            # Debugging: TP, FP, FN 값 확인
            # print(f"TP: {tp}, FP: {fp}, FN: {fn}")
            
            precision, recall = compute_precision_recall(tp, fp, fn)
            precisions.append(precision)
            recalls.append(recall)

        # 각 IoU threshold에 대한 Average Precision(AP)을 계산
        ap = compute_average_precision(precisions, recalls)
        all_precisions.append(ap)

    # mAP50 및 mAP5090을 계산
    map50 = all_precisions[0] if 0.5 in iou_thresholds else 0
    map5090 = np.mean(all_precisions)  # 모든 IoU threshold에 대한 평균

    return map50, map5090

def process_model_metrics(model_metrics, csv_output_path, plot_output_dir):
    
    with open(os.path.join(plot_output_dir, "results.pkl"), "wb") as f:
        pickle.dump(model_metrics, f, pickle.HIGHEST_PROTOCOL)
    
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)

    # 모델별 평균 계산
    average_results = {model_name: {metric: np.mean(values) for metric, values in metrics.items()} 
                       for model_name, metrics in model_metrics.items()}

    # pandas DataFrame으로 변환 후 CSV로 저장
    df = pd.DataFrame(average_results).T  # 모델 이름이 인덱스로 들어가도록 전치
    df.to_csv(csv_output_path, index=True)
    print(f"Average metrics CSV saved to {csv_output_path}")

    # 각 메트릭별로 그래프 생성 및 저장
    for metric in ['precision', 'recall', 'map50', 'map5090']:
        plt.figure(figsize=(8, 6))

        for model_name in average_results.keys():
            plt.bar(model_name, average_results[model_name][metric], label=model_name)

        plt.title(f'Model {metric.capitalize()} Averages')
        plt.xlabel('Model Name')
        plt.ylabel(metric.capitalize())
        plt.tight_layout()

        plot_path = os.path.join(plot_output_dir, f'{metric}_averages_plot.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"Plot saved to {plot_path}")

def draw_bboxes_on_image(image, boxes, scores=None, class_ids=None, class_names=None, color=(0, 255, 0), thickness=2):
    """
    이미지 위에 바운딩 박스를 그리는 함수
    - image: OpenCV로 읽어들인 이미지 배열
    - boxes: 바운딩 박스 좌표 리스트 (각각 [x1, y1, x2, y2])
    - scores: 신뢰도 점수 리스트 (optional)
    - class_ids: 클래스 ID 리스트 (optional)
    - class_names: 클래스 이름 리스트 (optional)
    - color: 바운딩 박스 색상 (BGR 튜플, 기본값은 초록색)
    - thickness: 바운딩 박스 두께
    """
    # 이미지 복사본 생성
    img_with_boxes = image.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  # 정수형으로 변환하여 좌표 추출
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        label = ""
        if class_ids is not None and class_names is not None:
            label += f"{class_names[class_ids[i]]}: "  # 클래스 이름 추가
        if scores is not None:
            label += f"{scores[i]:.2f}"  # 신뢰도 점수 추가

        # 라벨을 그린 위치에 텍스트 표시
        if label:
            cv2.putText(img_with_boxes, label, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img_with_boxes