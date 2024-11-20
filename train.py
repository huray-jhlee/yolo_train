import os
import json
import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

os.environ["OMP_NUM_THREADS"] = '8'

def train(args):
    # General Options 필터링
    general_args = {k: v for k, v in vars(args).items() if k in ["wandb", "save_dir", "resume_wandb_id"]}

    # Training Options 필터링
    training_args = {k: v for k, v in vars(args).items() if k in [
        "model", "data", "epochs", "batch", "device", "cache", 
        "workers", "project", "name", "box", "cls", "dfl", "resume"
    ]}
    
    # first, load augment args
    with open("config_aug.json", "r") as f:
        ModelConfig = json.load(f)
    
    # seccond, overwrite training args to argument args
    with open("config.json", "r") as f:
        config_args = json.load(f)
    
    # overwirte loss weight(dfl, box, cls), epoch, batch, data
    config_args.update(training_args)
    ModelConfig.update(config_args)
    general_args.update(ModelConfig)
    
    if args.resume_wandb_id is None:
        wandb.init(
            project=args.project,
            dir=args.save_dir,
            config=general_args
        )
    elif args.wandb :
        wandb.init(
            project=args.project,
            dir=args.save_dir,
            config=general_args,
            id = args.resume_wandb_id,
            resume="must"
        )
        
    
    if args.resume is None:
        model_path = ModelConfig["model"]
    else :
        model_path = args.resume
    
    model = YOLO(model_path, task="detect")
    
    if args.wandb:
        add_wandb_callback(
            model,
            enable_model_checkpointing=True
        )
    
    train_result = model.train(
        **ModelConfig
    )
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument("--wandb", action='store_true')
    general_group.add_argument("--save_dir", type=str, default="/data2/jh/detector/")
    general_group.add_argument("--resume_wandb_id", type=str, default=None)
    general_group.add_argument("--notes", type=str, default=None)
    
    training_group = parser.add_argument_group("Training Options")
    training_group.add_argument("--model", type=str, default="./models/yolov8n.pt")
    training_group.add_argument("--data", type=str, default="/data2/jh/exp4_data/241113_det4.yaml")
    training_group.add_argument("--epochs", type=int, default=10)
    training_group.add_argument("--batch", type=int, default=64)
    training_group.add_argument("--device", type=str, default="4,5")
    training_group.add_argument("--cache", type=str, default=None)
    training_group.add_argument("--workers", type=int, default=8)
    training_group.add_argument("--project", type=str, default="food_detector")
    training_group.add_argument("--name", type=str, default=None)
    training_group.add_argument("--box", type=float, default=8)
    training_group.add_argument("--cls", type=float, default=0.5)
    training_group.add_argument("--dfl", type=float, default=2)
    training_group.add_argument("--resume", type=str, default=None, help="resume model weight path")
    args = parser.parse_args()
    
    train(args)