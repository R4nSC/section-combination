import torch
import torch.nn as nn
from torchvision import models


class IntegratedEnsembleModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        # VGG16のモデルをロード
        if params.args.no_pretrained:
            self.network = models.vgg16(pretrained=False)  # 1からの学習
        else:
            self.network = models.vgg16(pretrained=True)  # fine-tuning

        # VGG16のネットワーク構成をリストとして保持
        vgg_lt = list(self.network.classifier)

        # VGG16の最終層の次元を変更する
        num_class = params.yaml[params.yaml['use_dataset']]['num_family']
        vgg_lt[6] = nn.Linear(4096, num_class, bias=True)
        self.network.classifier = nn.Sequential(vgg_lt[0], vgg_lt[1], vgg_lt[2], vgg_lt[3],
                                                vgg_lt[4], vgg_lt[5], vgg_lt[6])

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

        # VGG16のモデルをロード
        if params.args.no_pretrained:
            self.network = models.vgg16(pretrained=False)  # 1からの学習
        else:
            self.network = models.vgg16(pretrained=True)  # fine-tuning

        # VGG16のネットワーク構成をリストとして保持
        vgg_lt = list(self.network.classifier)

        # VGG16の最終層の次元を変更する
        num_class = params.yaml[params.yaml['use_dataset']]['num_family']
        vgg_lt[6] = nn.Linear(4096, num_class, bias=True)
        self.network.classifier = nn.Sequential(vgg_lt[0], vgg_lt[1], vgg_lt[2], vgg_lt[3],
                                                vgg_lt[4], vgg_lt[5], vgg_lt[6])

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
