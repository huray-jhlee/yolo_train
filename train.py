import os
import json
import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

os.environ["OMP_NUM_THREADS"] = '8'

def train(args):
    
    if args.resume_wandb_id is None:
        wandb.init(
            project=args.project,
            dir=args.save_dir,
            config=vars(args)
        )
    else :
        wandb.init(
            project=args.project,
            dir=args.save_dir,
            config=vars(args),
            id = args.resume_wandb_id,
            resume="must"
        )
        
    
    if args.resume is None:
        model_path = "/home/ai04/jh/codes/yolo_train/models/yolov8n.pt"
    else :
        model_path = args.resume
    
    model = YOLO(model_path)
    
    # Check args
    """
    add_wandb_callback(
        model: YOLO,
        epoch_logging_interval: int = 1,
        -> a 학습동안 prediction visualization 하는 인터벌
        enable_model_checkpointing: bool = False,
        -> a 아티팩트로 저장시키는 것... 은 해야지
        enable_train_validation_logging: bool = True,
        -> validation 데이터에 대한 예측과 GT이미지를 이미지 overlay형태로 제공
        -> wandb.Table로 .. mean-confidence와 per-class 수치를
        enable_validation_logging: bool = True,
        -> Only Validation
        enable_prediction_logging: bool = True,
        -> each prediction..
        max_validation_batches: Optional[int] = 1,
        -> 
        visualize_skeleton: Optional[bool] = True,
        -> use in pose estimation
    )
    """
    
    add_wandb_callback(
        model,
        enable_model_checkpointing=True
    )
    
    with open("config.json", "r") as f:
        ModelConfig = json.load(f)
    
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.gpus,
        cache=False if args.cache is None else args.cache,
        save_period=1,
        workers=args.workers,
        project=args.project,
        batch=args.batch,
        mosaic=ModelConfig["mosaic"],
        mixup=ModelConfig["mixup"],
        resume=False if args.resume is None else True,
        pretrained=True,
        patience=ModelConfig["patience"],
        warmup_epochs=ModelConfig["warmup_epochs"],
        lr0=ModelConfig["lr0"],
        lrf=ModelConfig["lrf"],
        cos_lr=ModelConfig["cos_lr"],
    )
    
    metrics = model.val()
    
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--data", type=str, default="/data/food_detector_dataset/food_dataset0927/food.yaml")
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--gpus", type=str)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="food_detector")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0)
    parser.add_argument("--save_dir", type=str, default="/data2/jh/detector/")
    parser.add_argument("--resume", type=str, default=None, help="resume model weight path")
    parser.add_argument("--resume_wandb_id", type=str, default=None)
    args = parser.parse_args()
    print(args.cache)
    
    train(args)