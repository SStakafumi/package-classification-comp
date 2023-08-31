# (1) initModel() で読み込むモデルとパラメータの形状が一致していることを確認

import argparse
import datetime
import os
import sys
import shutil
import hashlib

import numpy as np
import pandas as pd

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

IMAGE_SIZE = (256, 256)


class TestApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        # model path (必須)
        parser.add_argument('--model-path',
                            help='model path under models directory',
                            type=str,
                            )
        # epochごとparametersの保存先のpath (必須)
        parser.add_argument('--state-path',
                            help='state file path under models directory',
                            type=str,
                            )
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=16,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker process for background data loading',
                            default=8,
                            type=int,
                            )
        parser.add_argument('comment',
                            help='Comment suffix for wandb run.',
                            nargs='?',
                            default='SIGNATE',
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        assert self.cli_args.model_path, 'set your model_path like --model-path=resnet18!!'
        assert self.cli_args.state_path, 'set your state_path like --state-path=2023-08-30...'

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        log.info(
            f'Load model parameters from model/{self.cli_args.model_path}/{self.cli_args.state_path}')
        state_path = os.path.join(
            'models', self.cli_args.model_path, self.cli_args.state_path)
        self.state = torch.load(state_path)

        self.model = self.initModel()

    # (1)
    def initModel(self):
        model = ResNet18Wrapper(
            in_channels=3,
            pretrained=False
        )

        # load parameters
        log.info(' :adapt params to model')
        model.load_state_dict(self.state['model_state'])

        if self.use_cuda:
            log.info('Using CUDA; {} devices'.format(
                torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return model

    def initTestDl(self):
        test_ds = ImageDataset(dataType='test')
        batch_size = self.cli_args.batch_size

        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return test_dl

    def main(self):
        submit_df = pd.read_csv('data/sample_submit.csv',
                                names=['imgs_name', 'label_prob'])
        submit_df = submit_df.drop(['label_prob'], axis=1)

        probs_all, names_all = self.doTest()

        submit_df['imgs_name'] = names_all
        submit_df['label_probs'] = probs_all

        submit_df = submit_df.sort_values('imgs_name')

        # make directory
        file_path = os.path.join(
            'work', self.cli_args.model_path + '--' + self.cli_args.state_path)
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        # save submit file
        try:
            submit_df.to_csv(file_path[:-6] + '.csv',
                             index=False, header=None, mode='x')
        except FileExistsError:
            print('This submit csv file exists. Change file_name or delete exist file.')

    def doTest(self):
        probs_all = []
        names_all = []
        with torch.no_grad():
            self.model.eval()

            test_dl = self.initTestDl()

            for batch_tup in test_dl:
                imgs, names = batch_tup
                imgs_g = imgs.to(self.device, non_blocking=True)

                _, probs_g = self.model(imgs_g)

                probs_c = probs_g.to('cpu').detach().numpy()

                probs_all.append(probs_c)
                names_all.append(names)

            probs_all = np.concatenate(probs_all)
            names_all = np.concatenate(names_all)

            return probs_all[:, 1], names_all  # probality of 1


if __name__ == '__main__':
    TestApp().main()
