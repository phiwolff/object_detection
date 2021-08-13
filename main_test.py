
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

import shutil
import sys

from distutils.dir_util import copy_tree
from detectron2.modeling import build_model
from custom_mapper import Trainer



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


if __name__ == "__main__":

    for d in ["train", "val", "test"]:
        DatasetCatalog.register(d, lambda d=d: get_dicts(d),)
        MetadataCatalog.get(d).thing_classes = ["Acherontia atropos", "Smerinthus ocellata", "Mesembryhmus purpuralis"] 
    train_metadata = MetadataCatalog.get("train")  
    dicts = get_dicts("train")
    

    
    cfg = get_cfg()
    add_tensormask_config(cfg)
    cfg.merge_from_file("tensormask_R_50_FPN_6x.yaml")
    
    cfg.MODEL.WEIGHTS = "model_final_tensormask_pretrained_6x.pkl"
    
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.TENSOR_MASK.NUM_CLASSES = 3
    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # threshold entscheidet darüber, wie AP's ausfallen
    # => beachte unterschiede val nach train und val heier im Modelltest
    # prediction scores, die über diesem Wert liegen, werden als positive predictions gezählt,
    # die darunter als false predictions
    cfg.MODEL.TENSOR_MASK.SCORE_THRESH_TEST = 0.5
    #cfg.MODEL.TENSOR_MASK.NMS_THRESH_TEST = 0.001
    # das scheint absolut zu sein
    #cfg.MODEL.TENSOR_MASK.TOPK_CANDIDATES_TEST = 200
    cfg.TEST.DETECTIONS_PER_IMAGE = 270
    cfg.SOLVER.IMS_PER_BATCH_TEST = 1
    
    cfg.MODEL.TENSOR_MASK.NUM_CLASSES = 3
    
    cfg.DATALOADER.NUM_WORKERS = 1
   

     
    predictor = DefaultPredictor(cfg)
    
    train_metadata = MetadataCatalog.get("train")  
    dicts = get_dicts("train")



    evaluator = COCOEvaluator("test", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "test")
    cfg.DATASETS.TEST = ("test",)
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS)
    res = DefaultTrainer.test(cfg, model, evaluators=evaluator)
      
      
    im = cv2.imread("./images/Smerinthus_ocellata.jpg")
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
               metadata= train_metadata, 
               scale=0.8, 
               # remove the colors of unsegmented pixels
               instance_mode=ColorMode.IMAGE_BW   
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                                 
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1400,1400)
    cv2.imshow('image', v.get_image()[:, :, ::-1])
    cv2.waitKey(0)



    
 
     
  
 

    