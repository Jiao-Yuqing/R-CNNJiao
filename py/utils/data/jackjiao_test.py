import numpy  as np


import py
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from py.utils.util import parse_car_csv

#custom_finetune_dataset中的尝试
# annotation_path = "E:/R-CNN/py/data/classifier_car/train/Annotations/000091_1.csv"
# positive_rects=list()
# positive_sizes=list()
# rects = np.loadtxt(annotation_path, dtype=int, delimiter=' ')
# print(rects.shape[0])
# positive_rects.extend(rects)
# positive_sizes.append(len(rects))
# print(positive_rects)
# print(positive_sizes)
# print(len(positive_rects))

#finetune.py中加载预训练的模型有点问题
# import torchvision.models as models
# from torchvision.models import AlexNet_Weights
#
# model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
# print(model)

# print(rects.shape[1])

import py.utils.util
from py.utils.util import parse_xml
from py.utils.util import iou
from py.utils.util import compute_ious
target_box = parse_xml("E://R-CNNJiao//py//data//voc_car//val//Annotations//000060.xml")
target_box.shape#{ndarray:(5,4)}[[  1 137 427 333], [  1  62 479 235], [  1  27 422 141], [ 44   7 458 134], [199   6 475  98]]

pred_box = [25,6,300,200]

scores = iou(pred_box, target_box)


pass

