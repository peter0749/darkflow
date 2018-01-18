import os
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import jaccard_similarity_score, precision_recall_fscore_support
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
    real_cm_eye  = np.zeros((h,w))
    pred_cm_face = np.zeros((h,w))
    pred_cm_eye  = np.zeros((h,w))
    for xmin,xmax,ymin,ymax in y_true['face']:
        real_cm_face[ymin:(ymax+1), xmin:(xmax+1)] = 1.
    for xmin,xmax,ymin,ymax in y_true['re']:
        real_cm_eye[ymin:(ymax+1), xmin:(xmax+1)] = 1.
    for xmin,xmax,ymin,ymax in y_true['le']:
        real_cm_eye[ymin:(ymax+1), xmin:(xmax+1)] = 1.
    for ele in y_pred:
        xmax = ele['bottomright']['x']
        ymax = ele['bottomright']['y']
        xmin = ele['topleft']['x']
        ymin = ele['topleft']['y']
        if ele['label']=='face':
            pred_cm_face[ymin:(ymax+1), xmin:(xmax+1)] = 1.
        elif ele['label']=='re' or ele['label']=='le':
            pred_cm_eye[ymin:(ymax+1), xmin:(xmax+1)] = 1.
    face_score = jaccard_similarity_score(real_cm_face.flatten().astype(np.uint8), pred_cm_face.flatten().astype(np.uint8))
    face_precision, face_recall, face_f1, _ = precision_recall_fscore_support(real_cm_face.flatten().astype(np.uint8), pred_cm_face.flatten().astype(np.uint8))
    eye_score = jaccard_similarity_score(real_cm_eye.flatten().astype(np.uint8), pred_cm_eye.flatten().astype(np.uint8))
    eye_precision, eye_recall, eye_f1, _ = precision_recall_fscore_support(real_cm_eye.flatten().astype(np.uint8), pred_cm_eye.flatten().astype(np.uint8))
    return face_score, face_precision, face_recall, face_f1,
           eye_score, eye_precision, eye_recall, eye_f1


face_scores = []
face_precisions = []
face_recalls = []
face_f1s = []
eye_scores = []
eye_precisions = []
eye_recalls = []
eye_f1s = []
for file_name in os.listdir(TEST_DIR+'/jpg'):
    full_name_jpg = TEST_DIR+'/jpg/'+file_name
    full_name_xml = TEST_DIR+'/xml/'+file_name[:-3]+'xml'
    imgcv = cv2.imread(full_name_jpg)
    imglbl= read_lbl(full_name_xml)
    result = tfnet.return_predict(imgcv) ## a list
    face_score, face_precision, face_recall, face_f1, eye_score, eye_precision, eye_recall, eye_f1 = get_score(imglbl, result, imgcv.shape[0], imgcv.shape[1])
    print('face: %.2f, %.2f, %.2f, %.2f'%(face_score, face_precision, face_recall, face_f1))
    print('eye: %.2f, %.2f, %.2f, %.2f'%(eye_score, eye_precision, eye_recall, eye_f1))
    face_scores.append(face_score)
    face_precisions.append(face_precision)
    face_recalls.append(face_recall)
    face_f1s.append(face_f1)
    eye_scores.append(eye_score)
    eye_precisions.append(eye_precision)
    eye_recalls.append(eye_recall)
    eye_f1s.append(eye_f1)
print('mean face socre: %.2f%%'%(np.mean(face_scores)*100.))
print('mean face precision: %.2f%%'%(np.mean(face_precisions)*100.))
print('mean face recall: %.2f%%'%(np.mean(face_recalls)*100.))
print('mean face f1: %.2f%%'%(np.mean(face_f1s)*100.))
print('mean eye socre: %.2f%%'%(np.mean(eye_scores)*100.))
print('mean eye precision: %.2f%%'%(np.mean(eye_precisions)*100.))
print('mean eye recall: %.2f%%'%(np.mean(eye_recalls)*100.))
print('mean eye f1: %.2f%%'%(np.mean(eye_f1s)*100.))
