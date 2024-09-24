import os
import cv2
from tqdm import tqdm

from ultralytics import YOLO
from clearml import Task, Dataset

def train(args):
    model = YOLO("./yolov8n.pt")
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device="device=0"
    )
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--data", type=str, default="/data/jh/detection_dataset/240924_dataset/data.yaml")
    args = parser.parse_args()
    
    train(args)