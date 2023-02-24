import os
import numpy as np
import torch
from torch.utils import data
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils import ConfigParameter, make_logger, ImageFolderWithPaths
from ensemble_learning import train_and_evaluation_of_ensemble_model
from datasets import load_datasets

if __name__ == '__main__':
    params = ConfigParameter()
    logger = make_logger()

    params.yaml['use_dataset'] = params.args.dataset
    print(params.yaml['use_dataset'])

    if params.args.cross_validation:
        first_section_name = 'text'
        images_data = {}

        # 画像データセットを読み込む
        logger.info("Started loading the datasets.")
        datasets_section_list = params.yaml['ensemble_section_list'].copy()
        datasets_section_list.append('allSection')
        for section in datasets_section_list:
            dataset_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['ensemble'], section)
            images_data[section] = ImageFolderWithPaths(dataset_path, params.transforms)
        logger.info("Finished loading the datasets.")

        results = {}
        indices = {}

        # 層化分割によるK-Fold cross validation
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

            # Samplerを使ってデータローダを作成する
            sampler = {x: torch.utils.data.SubsetRandomSampler(indices[x])
                       for x in ['train', 'val', 'test']}

            data_loaders = {}
            data_sizes = {}
            for section in datasets_section_list:
                data_loaders[section] = {x: data.DataLoader(images_data[section],
                                                            batch_size=params.args.batch_size, sampler=sampler[x])
                                         for x in ['train', 'val', 'test']}
                data_sizes[section] = {x: len(indices[x])
                                       for x in ['train', 'val', 'test']}

            # アンサンブルモデルの学習および評価
            results[_fold] = train_and_evaluation_of_ensemble_model(params, logger, data_loaders, data_sizes)
            logger.info("Finished FOLD %s cross validation.", _fold + 1)

        # 交差検証の結果を出力する
        total = {}
        for key, value in results.items():
            for mode in params.yaml['all_model_mode']:
                print(f'Model {mode} Fold {key}: {value[mode]}')
                if key == 0:
                    total[mode] = value[mode]
                else:
                    total[mode] += value[mode]
                print(f'Model builtin-{mode} Fold {key}: {value[mode+10]}')
                if key == 0:
                    total[mode + 10] = value[mode + 10]
                else:
                    total[mode + 10] += value[mode + 10]
        for mode in params.yaml['all_model_mode']:
            print(f'\nModel{mode} Cross-Validation Average: {total[mode] / len(results.items())}')
            print(f'\nModel builtin-{mode} Cross-Validation Average: {total[mode + 10] / len(results.items())}')
    else:
        # .text/.rdata/.dataの3セクションをすべて持つサンプルからデータセットを作成
        logger.info("Started loading the datasets.")
        _dataset, data_loaders, data_sizes = load_datasets(params, mode=0)
        logger.info("Finished loading the datasets.")
        train_and_evaluation_of_ensemble_model(params, logger, data_loaders, data_sizes)
