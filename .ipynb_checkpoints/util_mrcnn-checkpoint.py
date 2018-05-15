import os
import math
import numpy as np
import cv2
from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn.visualize import display_instances 
from keras.preprocessing import image
import tensorflow as tf
from keras import backend as K

class Config_Car(Config):
    NAME = 'car'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7
    NUM_CLASSES     =81

config_car = Config_Car()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'model_data')

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_coco.h5') 

# Create model object in inference mode.
model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config_car)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True) 

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def car_mrcnn(input_img):    

    img = cv2.resize(input_img, (1024, 1024))
    img = image.img_to_array(img)
    results = model.detect([img], verbose=0)
    r = results[0]
    final_img = display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                  class_names, r['scores'])
    inp_shape = image.img_to_array(input_img).shape
    final_img = cv2.resize(final_img, (inp_shape[1], inp_shape[0]))
    
    return final_img

