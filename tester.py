import os
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from darkflow.net.build import TFNet
import cv2

TEST_DIR = './dataset/valid'
options = {
        "pbLoad": "built_graph/yolo-voc-afed.pb",
        "metaLoad": "built_graph/yolo-voc-afed.meta",
        "threshold": 0.5
        }
tfnet = TFNet(options)

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

def get_score(y_true, y_pred, h, w):
    real_cm_face = np.zeros((h,w))
    real_cm_re   = np.zeros((h,w))
    real_cm_le   = np.zeros((h,w))
    pred_cm_face = np.zeros((h,w))
    pred_cm_re   = np.zeros((h,w))
    pred_cm_le   = np.zeros((h,w))
    for xmin,xmax,ymin,ymax in y_true['face']:
        real_cm_face[ymin:(ymax+1), xmin:(xmax+1)] = 1.
    for xmin,xmax,ymin,ymax in y_true['re']:
        real_cm_re[ymin:(ymax+1), xmin:(xmax+1)] = 1.
    for xmin,xmax,ymin,ymax in y_true['le']:
        real_cm_le[ymin:(ymax+1), xmin:(xmax+1)] = 1.
    for ele in y_pred:
        xmax = ele['bottomright']['x']
        ymax = ele['bottomright']['y']
        xmin = ele['topleft']['x']
        ymin = ele['topleft']['y']
        if ele['label']=='face':
            pred_cm_face[ymin:(ymax+1), xmin:(xmax+1)] = 1.
        elif ele['label']=='re':
            pred_cm_re[ymin:(ymax+1), xmin:(xmax+1)] = 1.
        elif ele['label']=='le':
            pred_cm_le[ymin:(ymax+1), xmin:(xmax+1)] = 1.
    face_score = jaccard_similarity_score(real_cm_face.flatten().astype(np.uint8), pred_cm_face.flatten().astype(np.uint8))
    re_score = jaccard_similarity_score(real_cm_re.flatten().astype(np.uint8), pred_cm_re.flatten().astype(np.uint8))
    le_score = jaccard_similarity_score(real_cm_le.flatten().astype(np.uint8), pred_cm_le.flatten().astype(np.uint8))
    return face_score, re_score, le_score


face_scores = []
le_scores = []
re_scores = []
for file_name in os.listdir(TEST_DIR+'/jpg'):
    full_name_jpg = TEST_DIR+'/jpg/'+file_name
    full_name_xml = TEST_DIR+'/xml/'+file_name[:-3]+'xml'
    imgcv = cv2.imread(full_name_jpg)
    imglbl= read_lbl(full_name_xml)
    result = tfnet.return_predict(imgcv) ## a list
    face_score, le_score, re_score = get_score(imglbl, result, imgcv.shape[0], imgcv.shape[1])
    print('face score: %.2f'%(face_score))
    print('re score: %.2f'%(re_score))
    print('le score: %.2f'%(le_score))
    face_scores.append(face_score)
    re_scores.append(re_score)
    le_scores.append(le_score)
print('mean face socre: %.2f%%'%(np.mean(face_scores)*100.))
print('mean re socre: %.2f%%'%(np.mean(re_scores)*100.))
print('mean le socre: %.2f%%'%(np.mean(le_scores)*100.))
