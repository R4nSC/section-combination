import argparse
import torch
import torch.nn as nn
from torchvision import models


class Vgg16(nn.Module):
    def __init__(self, params: argparse.Namespace, model_path: str):
        """VGG16

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
            self.network = models.vgg16(pretrained=False)  # 1からの学習
        else:
            self.network = models.vgg16(pretrained=True)  # fine-tuning

        # VGG16のネットワーク構成をリストとして保持
        vgg_lt = list(self.network.classifier)

        # VGG16の最終層の次元を変更する
        vgg_lt[6] = nn.Linear(4096, params.yaml[params.yaml['use_dataset']]['num_family'], bias=True)
        self.network.classifier = nn.Sequential(vgg_lt[0], vgg_lt[1], vgg_lt[2], vgg_lt[3],
                                                vgg_lt[4], vgg_lt[5], vgg_lt[6])

        # 保存されたモデルパラメータを読み込む
        if params.args.model_use:
            self.network.load_state_dict(torch.load(model_path))

    # 順伝播
    def forward(self, x):
        x = self.network(x)
        return x
