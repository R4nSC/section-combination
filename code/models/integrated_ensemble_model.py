import torch
import torch.nn as nn

from .vgg import Vgg16
from .resnet import ResNet50


class IntegratedEnsembleModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        # VGG16 or ResNet50のモデルをロード
        if params.args.network == 'vgg16':
            self.network = Vgg16(params, self.load_parameter_path)
        else:
            self.network = ResNet50(params, self.load_parameter_path)

        num_class = params.yaml[params.yaml['use_dataset']]['num_family']

        # 次元を変更したモデルを2つ複製して合計3つのモデルを用意する
        self.network2 = self.network
        self.network3 = self.network

        self.linear = nn.Linear(num_class * 3, num_class, bias=True)

    # 順伝播
    def forward(self, x1, x2, x3):
        v1 = self.network(x1)
        v2 = self.network(x2)
        v3 = self.network(x3)

        bond_vector = torch.cat((v1, v2, v3), 1)
        y = self.linear(bond_vector)
        return y


class IntegratedEnsembleModelAddAllsection(nn.Module):
    def __init__(self, params):
        super().__init__()

        # VGG16 or ResNet50のモデルをロード
        if params.args.network == 'vgg16':
            self.network = Vgg16(params, self.load_parameter_path)
        else:
            self.network = ResNet50(params, self.load_parameter_path)

        num_class = params.yaml[params.yaml['use_dataset']]['num_family']

        # 次元を変更したモデルを3つ複製して合計4つのモデルを用意する
        self.network2 = self.network
        self.network3 = self.network
        self.network4 = self.network

        self.linear = nn.Linear(num_class * 4, num_class, bias=True)

    # 順伝播
    def forward(self, x1, x2, x3, x4):
        v1 = self.network(x1)
        v2 = self.network(x2)
        v3 = self.network(x3)
        v4 = self.network(x4)

        bond_vector = torch.cat((v1, v2, v3, v4), 1)
        y = self.linear(bond_vector)
        return y
