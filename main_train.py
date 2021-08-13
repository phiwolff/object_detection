
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
from detectron2.data import DatasetCatalog
import json
import os
import numpy as np
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from tensormask import add_tensormask_config

#from detectron2.projects.TensorMask.config import add_tensormask_config
#from projects.TensorMask.tensormask import config
from detectron2 import model_zoo
import train_net
from detectron2.utils.visualizer import ColorMode
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import time
import torch
import imgaug as ia
import imageio
from imgaug import augmenters as iaa
#from custom_mapper import custom_mapper
from custom_mapper import Trainer


import shutil
import sys

from distutils.dir_util import copy_tree
from detectron2.modeling import build_model



def get_dicts(json_name):

    img_dir = "./annotations/final/"
    json_file = os.path.join(img_dir, str(json_name) + ".json")

    
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    # loop through the entries in the JSON file
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        # add file_name, image_id, height and width information to the records
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]

        objs = []
        # one image can have multiple annotations, therefore this loop is needed
        for annotation in annos:
            # reformat the polygon information to fit the specifications
            anno = annotation["shape_attributes"]
            
            if anno["name"] == 'polygon':
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                tmp = annotation["region_attributes"]
                region_attributes = annotation["region_attributes"]["Klasse"]
    
            
            if "Acherontia atropos" in region_attributes:
                category_id = 0
            elif "Smerinthus ocellata" in region_attributes:
                category_id = 1
            elif "Mesembryhmus purpuralis" in region_attributes:
                category_id = 2
            

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
                "iscrowd": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
 
def startTraining():

    train_metadata = MetadataCatalog.get("train")  


    cfg = get_cfg()
    add_tensormask_config(cfg)
    cfg.merge_from_file("tensormask_R_50_FPN_6x.yaml")
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ()
    
    cfg.MODEL.WEIGHTS = "model_final_tensormask_pretrained_6x.pkl"

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.TENSOR_MASK.NUM_CLASSES = 3
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (512)   
    #cfg.MODEL.TENSOR_MASK.SCORE_THRESH_TEST = 0.9
    #cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 40000
    #cfg.SOLVER.WARMUP_METHOD = "linear"

    #cfg.TEST.EVAL_PERIOD = 5
    #cfg.MODEL.RPN.NMS_THRESH = 0.9
    #cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.9
    #cfg.TEST.DETECTIONS_PER_IMAGE = 300
    
    
    try:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)     
    except:
        shutil.rmtree(cfg.OUTPUT_DIR)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)
    

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    startEvaluation(cfg, trainer)

    
def startEvaluation(cfg, trainer):

    cfg.SOLVER.IMS_PER_BATCH_TEST = 1
    cfg.MODEL.TENSOR_MASK.NUM_CLASSES = 3

    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATASETS.TEST = ("val",)
    predictor = DefaultPredictor(cfg)

    
    evaluator = COCOEvaluator("val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "val")
    
    inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__ == "__main__":

    for d in ["train", "val", "test"]:
        DatasetCatalog.register(d, lambda d=d: get_dicts(d),)
        MetadataCatalog.get(d).thing_classes = ["Acherontia atropos", "Smerinthus ocellata", "Mesembryhmus purpuralis"]
 
     
        
    train_metadata = MetadataCatalog.get("train")  
    dicts = get_dicts("train")
    dict_test = get_dicts("test")

    
    startTraining()
    
    
    
  
    
    
    