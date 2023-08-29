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
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

random.seed(1)

warnings.filterwarnings("ignore")

IMAGE_SIZE = (256, 256)


@functools.lru_cache()  # foldのたびに呼ばれるのでメモ化
def getImageInfoList(fold_num):
    # dataをfolds分割して返す
    image_path_list = glob('/content/train/*.png')
    df = pd.read_csv(
        '/content/drive/MyDrive/MyStudy/MySIGNATE/package-classification-comp/data/train.csv')
    df = df.set_index('image_name')

    # データを格納するリスト
    images = []
    labels = []
    train_indices = []

    for img_path in image_path_list:
        # 画像読み込み
        img_name = os.path.split(img_path)[-1]  # 画像の名前 'xxxx.png'
        img = cv2.imread(img_path)  # numpy形式に変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(df.loc[img_name, 'label'])
        images.append(img)
        labels.append(label)

    skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=1)

    for (train_index, _) in skf.split(images, labels):
        train_indices.append(train_index)

    return images, labels, train_indices  # 全画像, ラベル, 訓練データのインデックス


ImageInfoTuple = namedtuple('ImageInfoTuple', 'image, label')


class ImageDataset(Dataset):
    def __init__(self, fold, fold_num, isTrain=True):
        images, labels, split_index = getImageInfoList(fold_num)

        # データをtrain, validで分ける
        self.imageInfo_list = []
        for i, (image, label) in enumerate(zip(images, labels)):
            if isTrain:
                if i in split_index[fold]:
                    self.imageInfo_list.append(ImageInfoTuple(image, label))
            else:
                if i not in split_index[fold]:
                    self.imageInfo_list.append(ImageInfoTuple(image, label))

    def image_transform(image, label):

        print(type(image))

        image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),
            transforms.Normalize((0.5433, 0.4686, 0.4039),
                                 (0.2455, 0.2476, 0.2497))  # 全訓練データで標準化
        ])

        image_t = image_transformer(image)
        image_t = image_t.unsqueeze(0)
        label_t = torch.tensor([not bool(label), bool(label)])
        return image_t, label_t

    def __len__(self):
        return len(self.imageInfo_list)

    def __getitem__(self, ndx):
        imageInfo_tup = self.imageInfo_list[ndx]
        image_t, label_t = self.image_transform(imageInfo_tup)
        return (image_t, label_t)
