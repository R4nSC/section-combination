import os
import sys
import torch
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split

sys.path.append('../')
from utils import ImageFolderWithPaths


# 各セクション単体画像データセットを別々で読み込む(mode=0)
def load_datasets_in_separate_sections(params):
    # 必要なdict型変数を用意しておく
    images_data = {}
    indices = {}
    datasets = {}
    data_loaders = {}
    data_sizes = {}

    # 'allSection'を含むセクションリストを用意する
    # ※元画像サンプル用のデータローダーも作成するため
    datasets_section_list = params.yaml['ensemble_section_list'].copy()
    datasets_section_list.append('allSection')

    for section_name in datasets_section_list:
        # アンサンブル学習用の単体画像のImageFolderを作成
        dataset_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['ensemble'], section_name)
        images_data[section_name] = ImageFolderWithPaths(dataset_path, params.transforms)

        # 層化分割(train:val:test -> 6:2:2)
        # trainとtrain以外に分割する
        indices['train'], indices['tmp'] = train_test_split(np.arange(len(images_data[section_name].targets)),
                                                            test_size=0.4,
                                                            stratify=images_data[section_name].targets,
                                                            random_state=0)
        # train以外をvalとtestに分割する
        indices['val'], indices['test'] = train_test_split(np.array(indices['tmp']),
                                                           test_size=0.5,
                                                           stratify=np.array(images_data[section_name].targets)
                                                           [indices['tmp']],
                                                           random_state=0)

        # train, val, testのサブセット，データローダーを作成
        datasets[section_name] = {x: data.Subset(images_data[section_name], indices=indices[x])
                                  for x in ['train', 'val', 'test']}
        data_loaders[section_name] = {x: data.DataLoader(datasets[section_name][x], batch_size=params.args.batch_size)
                                      for x in ['train', 'val', 'test']}
        data_sizes[section_name] = {x: len(datasets[section_name][x])
                                    for x in ['train', 'val', 'test']}

    return datasets, data_loaders, data_sizes


# 各セクション単体画像データセットを1つのローダとして読み込む(mode=1)
def load_datasets_as_single_loader(params):
    # 必要なdict型変数を用意しておく
    images_data = {}
    indices = {}
    malware_datasets = {}
    data_loaders = {}
    datasets_size = {}

    images_tensor = {}
    labels_tensor = {}
    # paths_tensor = {}

    ensemble_datasets = {}
    ensemble_dataloaders = {}
    ensemble_size = {}

    for section_name in params.yaml['ensemble_section_list']:
        # アンサンブル学習用の単体画像のImageFolderを作成
        dataset_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['ensemble'], section_name)
        images_data[section_name] = ImageFolderWithPaths(dataset_path, params.transforms)

        # 層化分割(train:val:test -> 6:2:2)
        # trainとtrain以外に分割する
        indices['train'], indices['tmp'] = train_test_split(np.arange(len(images_data[section_name].targets)),
                                                            test_size=0.4,
                                                            stratify=images_data[section_name].targets,
                                                            random_state=0)
        # train以外をvalとtestに分割する
        indices['val'], indices['test'] = train_test_split(np.array(indices['tmp']),
                                                           test_size=0.5,
                                                           stratify=np.array(images_data[section_name].targets)
                                                           [indices['tmp']],
                                                           random_state=0)

        # train, val, testのサブセット，データローダーを作成
        malware_datasets[section_name] = {x: data.Subset(images_data[section_name], indices=indices[x])
                                          for x in ['train', 'val', 'test']}
        data_loaders[section_name] = {x: data.DataLoader(malware_datasets[section_name][x], batch_size=params.args.batch_size)
                                      for x in ['train', 'val', 'test']}
        datasets_size[section_name] = {x: len(malware_datasets[section_name][x])
                                       for x in ['train', 'val', 'test']}

        # サブセットから画像とラベルのリストを作成する
        images_tensor[section_name] = {}
        for x in ['train', 'val', 'test']:
            # リストの初期化
            images_tensor[section_name][x] = []
            if section_name == 'text':
                labels_tensor[x] = []
                # paths_tensor[x] = tuple()

            # 画像とラベルのみを抽出してリストに追加する
            # ※ラベルの抽出はtextの時のみ
            for index in range(len(malware_datasets[section_name][x])):
                images_tensor[section_name][x].append(malware_datasets[section_name][x][index][0])
                if section_name == 'text':
                    labels_tensor[x].append(malware_datasets[section_name][x][index][1])
                    # paths_tensor[x] = paths_tensor[x] + (malware_datasets[section_name][x][index][2], )

            # 作成したリストをtensor型に変換する
            images_tensor[section_name][x] = torch.stack(images_tensor[section_name][x])
            if section_name == 'text':
                labels_tensor[x] = torch.tensor(labels_tensor[x], dtype=torch.int64)
                # paths_tensor[x] = torch.tensor(paths_tensor[x], dtype=torch.int64)

    # 3つの単体画像とそれらのラベルを含むデータセットを作成する
    # TODO: 3つの単体画像が同じサンプルかどうかチェックできていない（結果がそれなりになれば多分大丈夫）
    # all_images_tensor = {}
    # for x in ['train', 'val', 'test']:
    #     all_images_tensor[x] = {'text': images_tensor['text'][x],
    #                             'rdata': images_tensor['rdata'][x],
    #                             'data': images_tensor['data'][x]}

    ensemble_datasets = {x: data.TensorDataset(images_tensor['text'][x], images_tensor['rdata'][x],
                         images_tensor['data'][x], labels_tensor[x])
                         for x in ['train', 'val', 'test']}

    # 3つの単体画像とそれらのラベルを含むデータローダを作成する
    ensemble_dataloaders = {x: data.DataLoader(ensemble_datasets[x], batch_size=params.args.batch_size)
                            for x in ['train', 'val', 'test']}

    ensemble_size = {x: len(malware_datasets['text'][x])
                     for x in ['train', 'val', 'test']}

    return ensemble_datasets, ensemble_dataloaders, ensemble_size


# 指定されたモードに応じたデータセットを読み込んで返す
# ※読みこむのはtext/rdata/dataの3種
# mode=0: 各セクション単体画像データセットを別々で読み込む(allSectionも含む)
# mode=1: 各セクション単体画像データセットを1つのローダとして読み込む(allSectionは含まない)
def load_datasets(params, mode):
    if mode == 0:
        datasets, data_loaders, data_sizes = load_datasets_in_separate_sections(params)
    elif mode == 1:
        datasets, data_loaders, data_sizes = load_datasets_as_single_loader(params)

    return datasets, data_loaders, data_sizes
