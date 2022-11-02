import yaml
import argparse
import datetime
import torch
from torchvision import transforms


class ConfigParameter():
    """パラメータを管理するクラス
    """
    def __init__(self):
        # コマンドライン引数からのパラメータ
        self.args = self._parse_args()

        # yamlファイルから読み出す
        self.yaml = self._load_yaml('../yaml/config.yml')

        # そのほかに設定するパラメータ
        self.now_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _parse_args(self) -> argparse.Namespace:
        """コマンドライン引数によるパラメータ読み込み

        Returns
        -------
        argparse.Namespace
            読み込んだパラメータを保持したインスタンス
        """
        parser = argparse.ArgumentParser()

        parser.add_argument('-n', '--network', default='vgg16',
                            choices=['vgg16', 'resnet50'],
                            help='models network architecture (default: vgg)')
        parser.add_argument('-e', '--epochs', default=10, type=int,
                            help='number of epochs (default: 10)')
        parser.add_argument('-b', '--batch-size', default=100, type=int,
                            help='mini-batch size (default: 100)')
        parser.add_argument('-m', '--model-mode', default=0, type=int,
                            help='mode of ensemble model (default: 0)')
        parser.add_argument('--no-pretrained', action='store_true',
                            help='Flag for not using pre-trained model (default: False)')
        parser.add_argument('--gui', action='store_true',
                            help='Flag for GUI output (default: False)')
        parser.add_argument('--debug', action='store_true',
                            help='Flag for debugging output (default: False)')
        parser.add_argument('--model-use', action='store_true',
                            help='Flag for using saving model parameters (default: False')
        parser.add_argument('--model-save', action='store_true',
                            help='Flag for saving embedding model parameters (default: False')

        parser.add_argument('--result-analysis', action='store_true',
                            help='Flag for analysising of results (default: False')
        parser.add_argument('--cross-validation', action='store_true',
                            help='Flag for using cross validation (default: False')
        parser.add_argument('--ensemble-add-allsection', action='store_true',
                            help='Flag for add original model using allsection images (default: False')

        args = parser.parse_args()

        return args

    def _load_yaml(self, filepath: str) -> dict:
        """指定されたパスのyamlファイルからパラメータを読み込む

        Parameters
        ----------
        filename : str
            読み込みたいyamlファイルのパス

        Returns
        -------
        dict
            読み込んだパラメータの辞書
        """

        with open(filepath, 'r') as yml:
            config = yaml.safe_load(yml)

        return config
