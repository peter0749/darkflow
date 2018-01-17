import os
import xml.etree.ElementTree as ET
import numpy as np
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

from darkflow.net.build import TFNet
import cv2

TEST_DIR = './test'
options = {
        "pbLoad": "built_graph/yolo-voc-afed.pb",
        "metaLoad": "built_graph/yolo-voc-afed.meta",
        "threshold": 0.5
        }

def read_lbl(path):
    lbl = {'face': [], 're': [], 'le': []}
    root = ET.parse(str(path)).getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)
        lbl[name].append([xmin, xmax, ymin, ymax])
    return lbl

score_list = []
for file_name in os.listdir(TEST_DIR):
    full_name = TEST_DIR+'/'+file_name
    if full_name[:3]=='jpg':
        imgcv = cv2.imread(full_name)
        result = tfnet.return_predict(imgcv)
        print(result)
