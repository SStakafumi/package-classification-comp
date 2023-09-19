import math

from torch import nn as nn
# from torchvision.models import resnet18 # dropout なし
from resnet_dropout import resnet18  # dropout あり
from resnet_dropout import resnet50  # dropout あり


class ResNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # Batch normalization
        self.input_batchnorm = nn.BatchNorm2d(
            kwargs['in_channels'])

        # # ResNet18
        # self.resnet18 = resnet18(pretrained=kwargs['pretrained'])
        # self.resnet18.fc = nn.Linear(
        # in_features=1000, out_features=2, bias=True)  # 出力チャネル数を1000->2

        # ResNet50
        self.resnet50 = resnet50(pretrained=kwargs['pretrained'])
        self.resnet50.fc = nn.Linear(
            in_features=2048, out_features=2, bias=True)  # 出力チャネル数を1000->2

        # to probability
        self.head_softmax = nn.Softmax(dim=1)

        # initialize weights and bias
        if kwargs['pretrained'] == False:
            self._init_weights()

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.ConvTranspose2d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        # bn_output = self.input_batchnorm(input_batch)
        # # resnet18
        # resnet_output = self.resnet18(input_batch)

        # resnet50
        resnet_output = self.resnet50(input_batch)

        return resnet_output, self.head_softmax(resnet_output)
