import argparse
import datetime
import os
import sys

import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from .dsets import ImageDataset
from .model import ResNet18Wrapper

from util.logconf import logging


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# 評価マスク用の定数
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


class TrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=16,
                            type=int,
                            )
        parser.add_argument('--resnet-pretrained',
                            help='pretrain resnet model or not.',
                            default=True,
                            type=bool,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker process for background data loading',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=3,
                            type=int,
                            )
        parser.add_argument('--wandb-prefix',
                            default='first_test',
                            help='Data prefix to use for Weights and Biases',
                            )
        parser.add_argument('--fold-num',
                            help='Closs validation fold num',
                            default=4,
                            type=int,
                            )
        parser.add_argument('--finetune-params',
                            help='Update params when finetuning',
                            default=[],
                            type=list,
                            )
        parser.add_argument('comment',
                            help='Comment suffix for wandb run.',
                            nargs='?',
                            default='signate-comp',
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = ResNet18Wrapper(
            in_channels=3,
            pretrained=self.cli_args.resnet_pretrained,
        )

        # Fine tune : もしファインチューニングするパラメータが配列要素にあったら, そのパラメータ以外の重みの更新をOFF
        if self.cli_args.finetune_params:
            for name, param in model.named_parameters():
                if name not in self.cli_args.finetune_params:
                    param.required_grad = False

        if self.use_cuda:
            # log.info('Using CUDA; {} devices'.format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=1e-3, momentum=0.99)

    def initTrainDl(self, fold):
        train_ds = ImageDataset(
            fold, fold_num=self.cli_args.fold_num, isTrain=True)

        batch_size = self.cli_args.batch_size

        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self, fold):
        val_ds = ImageDataset(
            fold, fold_num=self.cli_args.fold_num, isTrain=False)

        batch_size = self.cli_args.batch_size

        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initWandB(self):
        pass

    def main(self):
        log.info('Starting {}, {}'.format(type(self).__name__, self.cli_args))
        self.totalFoldTrainSample_count = 0  # foldごとに何個のデータで学習したか

        # 交差検証
        for fold in range(self.cli_args.fold_num):
            # DataLoader
            train_dl = self.initTrainDl(fold=fold)
            val_dl = self.initValDl(fold=fold)

            for epoch_ndx in range(1, self.cli_args.epochs+1):
                # log
                log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    epoch_ndx,
                    self.cli_args.epochs,
                    len(train_dl),
                    len(val_dl),
                    self.cli_args.batch_size,
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

                # 1epoch内
                trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
                # 評価指標を記録
                self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(  # 評価マスクを初期化
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            'E{} Training'.format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,  # 何個目のバッチか
                batch_tup,  # (image, label)
                train_dl.batch_size,
                trnMetrics_g  # 評価マスク(これから埋める)
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalFoldTrainSample_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    # 1batchにおける平均損失を計算し、trnMetrics_gを埋める
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t = batch_tup  # (image, label)

        input_g = input_t.to(self.device, non_blocking=True)  # 新しくGPUに領域作る
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)  # Linear, Softmax
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(
            logits_g,
            label_g[:, 1],
        )

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        # 実際のラベル (0: 飲料, 1: 食料)
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1]
        metrics_g[METRICS_PRED_NDX,
                  start_ndx:end_ndx] = probability_g[:, 1]  # 食料であると予測した確率
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = loss_g  # 損失

        return loss_g.mean()  # バッチ平均した損失を返す

    def logMetrics(self, epoch_ndx, mode_str, metrics_t, classificationThreshold=0.5,):
        self.initWandB()

        log.info('E{} {}'.format(
            epoch_ndx,
            type(self).__name__
        ))
        # 以下bool型で評価値を示す(0: 飲料, 1: 食料だからちょっとめんどい)
        # 実際に飲料のマスク
        drinkLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        # 飲料であると予測したマスク
        drinkPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold
        # 実際に食料のマスク
        foodLabel_mask = ~drinkLabel_mask
        # 食料であると予測したマスク
        foodPred_mask = ~drinkPred_mask

        drink_count = int(drinkLabel_mask.sum())  # 実際の飲料データの数
        food_count = int(foodLabel_mask.sum())  # 実際の食料データの数

        trueDrink_count = neg_correct = int(
            (drinkLabel_mask & drinkPred_mask).sum())  # 実際に飲料で飲料と予測した数 (TN)
        trueFood_count = pos_correct = int(
            (foodLabel_mask & foodPred_mask).sum())  # 実際に食料で食料と予測した数 (TP)

        drinkButFood_count = fp_correct = drink_count - \
            neg_correct  # 実際は飲料だが食料と予測した数 (FP)
        foodButDrink_count = fn_correct = food_count - \
            pos_correct  # 実際は食料だが飲料と予測した数 (FN)

        metrics_dict = {}
        # 1epochの全データの損失
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/drink'] = metrics_t[METRICS_LOSS_NDX,
                                               drinkLabel_mask].mean()  # 飲料データの損失
        metrics_dict['loss/food'] = metrics_t[METRICS_LOSS_NDX,
                                              foodLabel_mask].mean()  # 食料データの損失
        # 全データのうち正しく分類できた割合
        metrics_dict['correct/all'] = (neg_correct +
                                       pos_correct) / metrics_t.shape[1] * 100
        # 飲料のうち正しく飲料と分類できた割合
        metrics_dict['correct/drink'] = (trueDrink_count) / drink_count * 100
        # 食料のうち正しく食料と分類できた割合
        metrics_dict['correct/food'] = (trueFood_count) / food_count * 100
        precision = metrics_dict['precision'] = trueFood_count / \
            np.float32(trueFood_count+fp_correct)
        recall = metrics_dict['recall'] = trueFood_count / \
            np.float32(trueFood_count+fn_correct)
        metrics_dict['f1_score'] = 2 * \
            (precision * recall) / (precision + recall)

        log.info((
            "E{} {:8} {loss/all:.4f} loss, "
            + "{correct/all:-5.1f}% correct, "
            + "{precision:.4f} precision, "
            + "{recall:.4f} recall, "
            + "{f1_score:.4f} f1 score"
        ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))


if __name__ == '__main__':
    TrainingApp().main()
