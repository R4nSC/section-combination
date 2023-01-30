import argparse
import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, params: argparse.Namespace, model_path: str):
        """ResNet50

        Parameters
        ----------
        params : argparse.Namespace
            読み込んだパラメータ
        model_path : str
            読み込むモデルのパラメータのパス
        """
        super().__init__()

        # VGG16のモデルをロード
        if params.args.no_pretrained:
            self.network = models.resnet50(pretrained=False)  # 1からの学習
        else:
            self.network = models.resnet50(pretrained=True)  # fine-tuning

        # ResNet50の最終層の次元を変更する
        self.network.fc = nn.Linear(self.network.fc.in_features, params.yaml[params.yaml['use_dataset']]['num_family'])

        # 保存されたモデルパラメータを読み込む
        if params.args.model_use:
            self.network.load_state_dict(torch.load(model_path))

    # 順伝播
    def forward(self, x):
        x = self.network(x)
        return x
