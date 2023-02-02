import os
import numpy as np
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils import ConfigParameter, make_logger, ImageFolderWithPaths
from datasets import load_datasets
from ensemble_learning import train_and_evaluation_of_ensemble_model

if __name__ == '__main__':
    params = ConfigParameter()
    logger = make_logger()

    if params.args.cross_validation:
        # 必要なdict型変数を用意しておく
        images_data = {}
        indices = {}
        malware_datasets = {}

        images_tensor = {}
        labels_tensor = {}

        ensemble_datasets = {}
        data_loaders = {}
        dataset_sizes = {}

        results = {}

        # 画像データセットを読み込む
        datasets_section_list = params.yaml['ensemble_section_list'].copy()
        if params.args.ensemble_add_allsection:
            datasets_section_list.append('allSection')

        logger.info("Started loading the datasets.")
        for section_name in datasets_section_list:
            dataset_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['ensemble'], section_name)
            images_data[section_name] = ImageFolderWithPaths(dataset_path, params.transforms)
        logger.info("Finished loading the datasets.")

        # 層化分割によるK-Fold cross validation
        first_section_name = 'text'
        n_splits = 5
        Kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        # K-Foldに従ってデータを2分割する（片方はテストデータにする）
        for _fold, (indices['train'], indices['test']) in enumerate(Kf.split(images_data[first_section_name], images_data[first_section_name].targets)):
            logger.info("Started FOLD %s cross validation.", _fold + 1)
            print(f'FOLD {_fold + 1}\n---')

            # トレーニングデータをさらに2分割する（片方はモデルの最適化等と行うための検証データにする）
            stratify = np.array(images_data[first_section_name].targets)[indices['train']]
            indices['train'], indices['val'] = train_test_split(np.array(indices['train']),
                                                                test_size=0.25, stratify=stratify)

            for section_name in datasets_section_list:
                # train, val, testのサブセットを作成する
                malware_datasets[section_name] = {x: data.Subset(images_data[section_name], indices=indices[x])
                                                for x in ['train', 'val', 'test']}

                # サブセットから画像とラベルのリストを作成する
                images_tensor[section_name] = {}
                for x in ['train', 'val', 'test']:
                    # リストの初期化
                    images_tensor[section_name][x] = []
                    if section_name == 'text':
                        labels_tensor[x] = []

                    # 画像とラベルのみを抽出してリストに追加する
                    # ※ラベルの抽出はtextの時のみ
                    for index in range(len(malware_datasets[section_name][x])):
                        images_tensor[section_name][x].append(malware_datasets[section_name][x][index][0])
                        if section_name == 'text':
                            labels_tensor[x].append(malware_datasets[section_name][x][index][1])

                    # 作成したリストをtensor型に変換する
                    images_tensor[section_name][x] = torch.stack(images_tensor[section_name][x])
                    if section_name == 'text':
                        labels_tensor[x] = torch.tensor(labels_tensor[x], dtype=torch.int64)

            # 単体画像とそれらのラベルを含むデータセットを作成する
            # TODO: 3つの単体画像が同じサンプルかどうかチェックできていない（結果がそれなりになれば多分大丈夫）
            if params.args.ensemble_add_allsection:
                ensemble_datasets = {x: data.TensorDataset(images_tensor['allSection'][x], images_tensor['text'][x],
                                    images_tensor['rdata'][x], images_tensor['data'][x], labels_tensor[x])
                                    for x in ['train', 'val', 'test']}
            else:
                ensemble_datasets = {x: data.TensorDataset(images_tensor['text'][x], images_tensor['rdata'][x],
                                    images_tensor['data'][x], labels_tensor[x])
                                    for x in ['train', 'val', 'test']}

            # 3つの単体画像とそれらのラベルを含むデータローダを作成する
            data_loaders = {x: data.DataLoader(ensemble_datasets[x], batch_size=params.args.batch_size)
                                    for x in ['train', 'val', 'test']}

            dataset_sizes = {x: len(malware_datasets['text'][x])
                             for x in ['train', 'val', 'test']}

            # アンサンブルモデルの学習および評価
            results[_fold] = train_and_evaluation_of_ensemble_model(params, logger, data_loaders, dataset_sizes)
            logger.info("Finished FOLD %s cross validation.", _fold + 1)

        # 交差検証の結果を出力する
        total = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value}')
            total += value

        print(f'\nCross-Validation Average: {total / len(results.items())}')
    else:
        # .text/.rdata/.dataの3セクションをすべて持つサンプルからデータセットを作成
        logger.info("Started loading the datasets.")
        _dataset, data_loaders, dataset_sizes = load_datasets(params, mode=1)
        logger.info("Finished loading the datasets.")

        logger.info("Started training integrated ensemble model.")
        train_and_evaluation_of_ensemble_model(params, logger, data_loaders, dataset_sizes)
        logger.info("Finished training integrated ensemble model.")
