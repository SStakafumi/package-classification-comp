import argparse
import datetime
import os
import sys
import shutil
import hashlib

import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from torch.utils.tensorboard import SummaryWriter

from util.util import enumerateWithEstimate
from dsets import ImageDataset
from model import ResNet18Wrapper

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
        parser.add_argument('--pretrained',
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
                            default=1,
                            type=int,
                            )
        parser.add_argument('--prefix',
                            default='test',
                            help='Data prefix to use for Weights and Biases',
                            )
        parser.add_argument('--finetune-params',
                            help='Update params when finetuning',
                            default=['resnet18.fc.weight', 'resnet18.fc.bias'],
                            type=list,
                            )
        parser.add_argument('--validation-cadence',  # valの間隔 (単位: epoch)
                            help='Interval at which verification is performed',
                            default=1,
                            type=int,
                            )
        parser.add_argument('--learning-rate',
                            help='model optimizer learning rate',
                            default=1e-3,
                            type=float,
                            )
        parser.add_argument('comment',
                            help='Comment suffix for wandb run.',
                            nargs='?',
                            default='SIGNATE',
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = ResNet18Wrapper(
            in_channels=3,
            pretrained=self.cli_args.pretrained,
        )

        # Fine tune : もしファインチューニングするパラメータが配列要素にあったら, そのパラメータ以外の重みの更新をOFF
        if self.cli_args.finetune_params:
            for name, param in model.named_parameters():
                if name not in self.cli_args.finetune_params:
                    param.required_grad = False

        if self.use_cuda:
            log.info('Using CUDA; {} devices'.format(
                torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=self.cli_args.learning_rate, momentum=0.99)
        # return Adam(self.model.parameters())

    def initTrainDl(self):
        train_ds = ImageDataset(isTrain=True)
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

    def initValDl(self):
        val_ds = ImageDataset(isTrain=False)
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

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join(
                'vis', self.cli_args.prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls' + self.cli_args.comment)

    def main(self):
        log.info('Starting {}, {}'.format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        self.validation_cadence = self.cli_args.validation_cadence

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

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                # if val is wanted
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)  # AUC

                self.saveModel(epoch_ndx, score == best_score)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(  # initialize metirics mask
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
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

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                'E{} Validation'.format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )

            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                )

        return valMetrics_g.to('cpu')

    # 1batchにおける平均損失を計算し、trnMetrics_gを埋める
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t = batch_tup  # (image, label)

        input_g = input_t.to(self.device, non_blocking=True)  # 新しくGPUに領域作る
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)  # fc, Softmax

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
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g  # 損失

        return loss_g.mean()  # バッチ平均した損失を返す

    def logMetrics(self, epoch_ndx, mode_str, metrics_t, classificationThreshold=0.5):
        self.initTensorboardWriters()

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
        # 飲料データの損失
        metrics_dict['loss/drink'] = metrics_t[METRICS_LOSS_NDX,
                                               drinkLabel_mask].mean()
        # 食料データの損失
        metrics_dict['loss/food'] = metrics_t[METRICS_LOSS_NDX,
                                              foodLabel_mask].mean()
        # AUC
        metrics_dict['AUC'] = roc_auc_score(
            (metrics_t[METRICS_LABEL_NDX]).detach().numpy().copy(), metrics_t[METRICS_PRED_NDX].detach().numpy().copy())
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
            + "{correct/all:-5.1f}% correct/all, "
            + "{correct/drink:-5.1f}% correct/drink, "
            + "{AUC:.4f} AUC, "
            + "{precision:.4f} precision, "
            + "{recall:.4f} recall, "
            + "{f1_score:.4f} f1 score"
        ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))

        writer = getattr(self, mode_str+'_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        # bins = [x/50.0 for x in range(51)]

        # negHist_mask = drinkLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        # posHist_mask = foodLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        # if negHist_mask.any():
        #     writer.add_histogram(
        #         'is_neg',
        #         metrics_t[METRICS_PRED_NDX, negHist_mask],
        #         self.totalTrainingSamples_count,
        #         bins=bins,
        #     )
        # if posHist_mask.any():
        #     writer.add_histogram(
        #         'is_pos',
        #         metrics_t[METRICS_PRED_NDX, posHist_mask],
        #         self.totalTrainingSamples_count,
        #         bins=bins,
        #     )

        return metrics_dict['AUC']

    def saveModel(self, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'models',
            self.cli_args.prefix,
            '{}_{}_{}.state'.format(
                self.time_str,  # 学習開始時間
                self.cli_args.comment,  # defatult: SIGNATE
                self.totalTrainingSamples_count,  # 全学習データ数
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        # 保存データ
        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),  # parameter
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }

        # 保存
        torch.save(state, file_path)

        log.info('Saved model params to {}'.format(file_path))

        if isBest:
            best_path = os.path.join(
                'models',
                self.cli_args.prefix,
                f'{self.time_str}_{self.cli_args.comment}.best.state'
            )
            # bestなパラメータをコピーして上書き保存
            shutil.copyfile(file_path, best_path)

            log.info('Saved model params to {}'.format(best_path))

        with open(file_path, 'rb') as f:
            log.info('SHA1: ' + hashlib.sha1(f.read()).hexdigest())


if __name__ == '__main__':
    TrainingApp().main()
