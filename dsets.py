import copy
import csv
import functools
import math
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import warnings
import random
import cv2
from collections import namedtuple
from sklearn.model_selection import train_test_split
from torchvision import transforms

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

random.seed(1)

warnings.filterwarnings("ignore")

IMAGE_SIZE = (256, 256)


@functools.lru_cache(1)
def getImageInfoList():
    image_path_list = glob('/content/train/*.png')
    df = pd.read_csv(
        '/content/drive/MyDrive/MyStudy/MySIGNATE/package-classification-comp/data/train.csv')
    df = df.set_index('image_name')

    # データを格納するリスト
    imgs = []
    labels = []

    for img_path in image_path_list:
        # 画像読み込み
        img_name = os.path.split(img_path)[-1]  # 画像の名前 'xxxx.png'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        label = int(df.loc[img_name, 'label'])

        imgs.append(img)
        labels.append(label)

    return imgs, labels


def data_transformer(img, label):

    image_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        transforms.Normalize((0.5433, 0.4686, 0.4039),
                             (0.2455, 0.2476, 0.2497))  # 全訓練データで標準化
    ])

    image_t = image_transformer(img)

    label_t = torch.tensor([not bool(label), bool(label)], dtype=torch.long)
    return image_t, label_t


class ImageDataset(Dataset):
    def __init__(self, isTrain=True):
        imgs, labels = getImageInfoList()

        # データをtrain, validで分ける
        trn_imgs, val_imgs, trn_lables, val_labels = train_test_split(
            imgs, labels, train_size=0.75, stratify=labels, random_state=1)

        if isTrain:
            self.imgs = trn_imgs
            self.labels = trn_lables
        else:
            self.imgs = val_imgs
            self.labels = val_labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, ndx):
        img = self.imgs[ndx]
        label = self.labels[ndx]
        image_t, label_t = data_transformer(img, label)
        return (image_t, label_t)
