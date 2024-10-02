import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO


def train(args):
    
    wandb.init(project="food_detector",
               config={
                   "epochs": args.epochs,
                   "batch_size": args.batch,
                   "resolution": args.imgsz,
                   "gpus": args.gpus
               })
    
    model = YOLO("./models/yolov8n.pt")
    
    add_wandb_callback(model, enable_model_checkpointing=True)
    
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.gpus,
        cache=False if args.cache is None else args.cache,
        save_period=1,
        workers=args.workers,
        project="food_detector"
    )
    
    metrics = model.val()
    
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--data", type=str, default="/data/food_detector_dataset/food_dataset0927/food.yaml")
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--gpus", type=str)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(args.cache)
    
    train(args)