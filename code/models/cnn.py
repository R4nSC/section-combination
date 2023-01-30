from .vgg import Vgg16
from .resnet import ResNet50


class CNN():
    def __init__(self, params, section=None):
        self.section = section
        self.load_parameter_path = self.get_model_path(params, section, mode=0)
        self.write_parameter_path = self.get_model_path(params, section, mode=1)
        self.model = self.select_models(params)

    # 引数によって指定されたモデルを返す
    def select_models(self, params):
        # 指定されたモデルを読み込む
        if params.args.network == 'vgg16':
            model = Vgg16(params, self.load_parameter_path)
        else:
            model = ResNet50(params, self.load_parameter_path)

        # 計算デバイスを指定(CPU or GPU)
        model = model.to(params.device)

        # デバッグモードならモデルのネットワーク構成を標準出力
        if params.args.debug:
            print(model)

        return model

    # 条件に合ったモデルパラメータのパスを返す
    # パスの形式:= {セクション名}_{モデル名}_{エポック数}_{バッチサイズ}_ver{バージョン}.pth
    def get_model_path(self, params, section, mode=0):
        # 読み込み用(mode=0)か書き込み用(mode≠0)に応じて，操作するファイルのバージョンを変える
        # 読み込み用は指定されたバージョン，書き込み用は指定されたバージョンの次
        # TODO: 読み込み用は指定されたバージョンで，書き込み用はまだ存在しないバージョン(最新版)に書き込むようにしたい
        file_version = params.yaml['model_dir']['file_version']
        if mode != 0:
            file_version = file_version + 1

        # セクション名が指定されていたらファイル名に含める
        model_filename = f'{params.args.network}_e{params.args.epochs}_b{params.args.batch_size}_ver{file_version}.pth'
        if section is not None:
            model_filename = f'{section}_{model_filename}'

        # モデルパラメータ用のディレクトリパスを繋げる
        model_path = f'{params.yaml["model_dir"]["model"]}/{model_filename}'
        return model_path
