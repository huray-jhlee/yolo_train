from ultralytics import YOLO
from PIL import Image
import cv2
from ultralytics.utils.plotting import Annotator
import os
import csv
import re
import argparse
import shutil
from tqdm import tqdm
import random
import torch

def copy_files(src, dst):
    try:
        shutil.copy(src, dst)
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")


def createDir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Failed to create the directory.")


def train(ModelConfig):
    model = YOLO(ModelConfig['model_path'])
    results = model.train(data=ModelConfig['data_path'], project = ModelConfig['project'], name = ModelConfig['name'],
                        pretrained = ModelConfig['pretrained'], device = ModelConfig['device'],
                        epochs = ModelConfig['epoch'], patience = ModelConfig['patience'], warmup_epochs = ModelConfig['warmup_epochs'],
                        imgsz=ModelConfig['input_size'], batch = ModelConfig['batch'], single_cls = ModelConfig['single_cls'], 
                        lr0=ModelConfig['lr0'], lrf = ModelConfig['lrf'], optimizer = ModelConfig['optimizer'], cos_lr = ModelConfig['cos_lr'], workers=8, 
                        cache = False, resume = False)
    
    metrics = model.val()



if __name__ == '__main__':

    # 1. dataset parsing
    dataset_path = '/home/ai01/food_dataset6/' 
    data_dict = {}
    data_dict['huray_food_detector'] = []

    # classfile = open('datasets/food-101/meta/classes.txt')
    # class_list = []
    # lines = classfile.readlines()
    # for line in lines:
    #     class_list.append(str(line.strip()))
    # classfile.close()

    class_list = ['Food', 'Processed']
    data_generated = False
    
    if data_generated is False:
        train_image_path = '/home/ai01/food_dataset6/train/images'
        val_image_path = '/home/ai01/food_dataset6/val/images'
        train_list_txt = open(os.path.join(dataset_path, 'train.txt'), 'w')
        val_list_txt = open(os.path.join(dataset_path, 'val.txt'), 'w')
        
        for file in os.listdir(train_image_path):
            if os.path.isfile(os.path.join(train_image_path, file)):
                ext = os.path.splitext(file)[-1]
                if ext == '.jpg' or ext == '.JPEG' or ext == '.PNG':
                    train_list_txt.write(os.path.join(train_image_path, file))
                    train_list_txt.write('\n')
            
        for file in os.listdir(val_image_path):
            if os.path.isfile(os.path.join(val_image_path, file)):
                ext = os.path.splitext(file)[-1]
                if ext == '.jpg' or ext == '.JPEG' or ext == '.PNG':
                    val_list_txt.write(os.path.join(val_image_path, file))
                    val_list_txt.write('\n')
        
        train_list_txt.close()
        val_list_txt.close()
        
# multi-GPU를 사용하기 위해서는 반드시 terminal에서 export MKL_SERVICE_FORCE_INTEL=TRUE 를 먼저 실행할 것!!
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "6"
device_ids = "6"
# torch.set_num_threads(1)


# 3. train
# model = YOLO("model/yolov8l-cls.pt")
# results = model.train(data='dataset', epochs=10, imgsz=128, device='cpu', batch=8)
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default = 'model/food_best.pt', type = str, required = True)
parser.add_argument('--data_path', default = '/home/ai01/food_dataset/food.yaml', type = str, required = True)
parser.add_argument('--project_name', type = str, required = True)
# parser.add_argument('--pretrained', type = bool, action = 'store_true')
parser.add_argument('--device_ids', type = str, required = True)
parser.add_argument('--epoch', type = int, required = True)
parser.add_argument('--batch', type = int, required = True)
parser.add_argument('--lr0', type = float, required = True)
parser.add_argument('--optimizer', type = str, required = True)
# parser.add_argument('--cos_lr', type = bool, action = 'store_true')
# args = parser.parse_args()

ModelConfig = {
        'data_path': '/home/ai01/food_dataset6/food.yaml',
        'patience': 0,
        'warmup_epochs': 2,
        'input_size': 640,
        'single_cls': False,
        'lrf': 0.01,
        }

# ModelConfig['model_path'] = args.model_path
# ModelConfig['data_path'] = args.data_path
# ModelConfig['project_name'] = args.project_name
# ModelConfig['name'] = f'{os.path.basename(args.model_path)}_{args.batch}_{args.lr0}_{args.optimizer}'
# ModelConfig['pretrained'] = args.pretrained
# ModelConfig['device'] = [id for id in args.device_ids.split(',')]
# ModelConfig['epoch'] = args.epoch
# ModelConfig['batch'] = args.batch
# ModelConfig['lr0'] = args.lr0
# ModelConfig['optimizer'] = args.optimizer
# ModelConfig['cos_lr'] = args.cos_lr

ModelConfig['project'] = 'food_detect_test'
# ModelConfig['model_path'] = '/home/ai01/project/active_learning_test/food_detect_test/best.pt_64_0.01_auto4/weights/best.pt'
ModelConfig['model_path'] = 'model/food_best.pt'
# ModelConfig['cfg'] = '/data3/food_dataset/food.yaml'
# ModelConfig['data_path'] = '/data3/food_dataset/food.yaml'
# ModelConfig['data_path'] = 'food-101.yaml'
# ModelConfig['project_name'] = 'learning test'
ModelConfig['pretrained'] = True
ModelConfig['device'] = device_ids
ModelConfig['epoch'] = 15
ModelConfig['batch'] = 32
ModelConfig['lr0'] = 0.01
ModelConfig['optimizer'] = 'auto'
ModelConfig['cos_lr'] = False

model_path = ModelConfig['model_path']
batch = ModelConfig['batch']
lr0 = ModelConfig['lr0']
optimizer = ModelConfig['optimizer']

ModelConfig['name'] = f'{os.path.basename(model_path)}_{batch}_{lr0}_{optimizer}'

train(ModelConfig)

# 4. extract confidence value from test data
    

# 5. remake training labeling txt (add low confidence data)


# 6. retrain
    
# 7. test
    




