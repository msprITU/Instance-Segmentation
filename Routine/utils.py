
import sys
import os
import math
import random
import numpy as np
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings

import datetime
import cv2
from itertools import groupby
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology



def DateCaptured():
    dt = str(datetime.datetime.now())
    date, time = dt.split()
    time = time.split(".")[0]
    return date+" "+time

def coco_bbox_creator(x, y):
    x = list(map(lambda x: float(x), x))
    y = list(map(lambda x: float(x), y))
    x_min = min(x)
    y_min = min(y)
    w = max(x) - x_min
    h = max(y) - y_min
    return [x_min, y_min, w, h]

def iou_calculator(bbox1, bbox2):
    """
    Parameters:   bbox1, bbox2: list or numpy array of bounding box coordinates.
    The input should contain the top-left corner's x and y coordinates and 
    width and height of the bounding boxes.
    
    Assertations: width and height informations of bbox1 and bbox2 should be 
    larger than 0.
    
    Returns:      iou: A floating point decimal representing the IoU ratio, which
    is the division of bounding box areas of intersection to their union.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    assert w1 and w2 > 0
    assert w1 and h2 > 0
    
    iou = 0
    if (((x1>x2 and x1<x2+w2) or (x1+w1>x2 and x1+w1<x2+w2) or 
        (x2>x1 and x2<x1+w1) or (x2+w2>x1 and x2+w2<x1+w1)) and 
        ((y1>y2 and y1<y2+h2) or (y1+h1>y2 and y1+h1<y2+h2) or
        (y2>y1 and y2<y1+h1) or (y2+h2>y1 and y2+h2<y1+h1))):
        iou_xmin = float(max(x1, x2))
        iou_xmax = float(min(x1+w1, x2+w2))
        iou_ymin = float(max(y1, y2))
        iou_ymax = float(min(y1+h1, y2+h2))
        intersection_area = (iou_ymax - iou_ymin)*(iou_xmax - iou_xmin)
        total_area = float(w1)*float(h1) + float(w2)*float(h2) - intersection_area
        iou = intersection_area/total_area
    return iou

def txt_bbox_parser(input_location):
    """
    Parameters: input_location: The input will be a text file denoting the bounding
    boxes for every frame in such format:
    frame_id, x1, y1, x2, y2, x3, y3, x4, y4
    
    Returns: image_nr: A list containing the the frame numbers. 
             xywh: A list containing the bounding boxes in (x1, y1, w, h) format,
                   where w and h are the width and height of a bounding box respectively.
    """
    with open(input_location) as f:
        bboxes = f.readlines()
    image_nr = []
    xywh = []
    for d, bbox in enumerate(bboxes):
        image_nr.append(bbox.split(",")[0])
        coords = np.array(bbox.split(",")[1:]).reshape((-1, 2))
        x, y = coords[:,0], coords[:,1]
        xywh.append(coco_bbox_creator(x, y))
    return image_nr, xywh

def mask2poly(mask):
    _, mask = cv2.threshold(mask,1,1, cv2.THRESH_BINARY)  #threshold binary image
    mask = binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, min_size=60)
    mask = np.array(mask, np.uint8)
    _,countours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for countour in countours:
        if countour.size >=6:
            polygons.append(countour.flatten().tolist())
    return polygons, mask

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))
    return rle
# Some functions to be used:
def isjpg(string):
    if string[-4:]==".jpg":
        return True 
    
def ispng(string):
    if string[-4:]==".png":
        return True
    
def save_bboxes(image_name, bbox_coor):
    with open('img_bbox.txt','a+') as file:
        file.write(str(image_name)+','+str(bbox_coor[0])+','+str(bbox_coor[1])+','+str(bbox_coor[2])+','+str(bbox_coor[3])+'\n')

def xml_retreiver(tree_object, key_name):
    for iterator in tree_object.iter(key_name):
        return iterator.text
