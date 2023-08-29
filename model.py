import math

from torch import nn as nn
from torchvision.models import resnet18


class ResNet18Wrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # batch正規化
        self.input_batchnorm = nn.BatchNorm2d(
            kwargs['in_channels'])

        # ResNet18
        self.resnet18 = resnet18(pretrained=kwargs['pretrained'])
        self.resnet18.fc = nn.Linear(
            in_features=512, out_features=2, bias=True)  # 出力チャネル数を1000->2

        # 確率変換
        self.head_softmax = nn.Softmax(dim=1)

        # 重みの初期化(nn.Linearのみ, ResNetの重みが初期化されちゃう？)
        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.kaiming_nomal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        # bn_output = self.input_batchnorm(input_batch)
        resnet_output = self.resnet18(input_batch)

        return resnet_output, self.head_softmax(resnet_output)
