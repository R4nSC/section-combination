import torch
import torch.nn as nn

from .vgg import Vgg16
from .resnet import ResNet50


class IntegratedEnsembleModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        # VGG16 or ResNet50のモデルをロード
        if params.args.network == 'vgg16':
            self.network = Vgg16(params, "")
            self.network2 = Vgg16(params, "")
            self.network3 = Vgg16(params, "")
        else:
            self.network = ResNet50(params, "")
            self.network2 = ResNet50(params, "")
            self.network3 = ResNet50(params, "")

        num_class = params.yaml[params.yaml['use_dataset']]['num_family']

        print(self.network)

        self.linear = nn.Linear(num_class * 3, num_class, bias=True)

    # 順伝播
    def forward(self, x1, x2, x3):
        v1 = self.network(x1)
        v2 = self.network2(x2)
        v3 = self.network3(x3)

        bond_vector = torch.cat((v1, v2, v3), 1)
        y = self.linear(bond_vector)
        return y


class IntegratedEnsembleModelAddAllsection(nn.Module):
    def __init__(self, params):
        super().__init__()

        # VGG16 or ResNet50のモデルをロード
        if params.args.network == 'vgg16':
            self.network = Vgg16(params, "")
            self.network2 = Vgg16(params, "")
            self.network3 = Vgg16(params, "")
            self.network4 = Vgg16(params, "")
        else:
            self.network = ResNet50(params, "")
            self.network2 = ResNet50(params, "")
            self.network3 = ResNet50(params, "")
            self.network4 = ResNet50(params, "")

        num_class = params.yaml[params.yaml['use_dataset']]['num_family']

        print(self.network)

        self.linear = nn.Linear(num_class * 4, num_class, bias=True)

    # 順伝播
    def forward(self, x1, x2, x3, x4):
        v1 = self.network(x1)
        v2 = self.network2(x2)
        v3 = self.network3(x3)
        v4 = self.network4(x4)

        bond_vector = torch.cat((v1, v2, v3, v4), 1)
        y = self.linear(bond_vector)
        return y
