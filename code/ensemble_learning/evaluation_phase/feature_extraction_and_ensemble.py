import os
import sys
import optuna
import logging
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.append('../')
from utils import objective_variable

# 3つのモデルの最終全結合層の出力を特徴量として抽出する
def feature_extraction_by_DLmodel(params, feature_model, data_loaders, data_mode='train'):
    # 各テストデータに対する予測とラベルを保持するdict型を確保する
    all_pred_by_section = {}
    all_label_by_section = {}
    all_path_by_section = {}

    # セクション単体画像で学習した各モデルでテストデータを予測した結果を取得する
    # TODO: この辺りは関数化できるかもしれないね
    for section_name in params.yaml['ensemble_section_list']:
        # 学習済みモデルをテストモードにする
        model = feature_model[section_name].model
        model.eval()
        all_pred_by_section[section_name] = torch.zeros(1)
        all_label_by_section[section_name] = torch.zeros(1)
        all_path_by_section[section_name] = tuple()

        # CPUモードとGPUモードへの切り替え
        all_pred_by_section[section_name] = all_pred_by_section[section_name].to(params.device)
        all_label_by_section[section_name] = all_label_by_section[section_name].to(params.device)

        with torch.no_grad():
            output_flg = True
            # PyTorchのシード値を固定にする(各セクション間で同じサンプルが同じ順序で出てくるはず...)
            torch.manual_seed(42)
            for datas in data_loaders[section_name][data_mode]:
                images, labels, paths = datas

                # CPUモードとGPUモードへの切り替え
                images = images.to(params.device)
                labels = labels.to(params.device)

                # 学習済みモデルに画像を入力した出力を入手する
                preds = model(images)

                # (Debug)モデルからの各出力を一度だけ出力する
                if output_flg:
                    # 最初だけは出力とラベルをそのまま突っ込む
                    output_flg = False
                    all_pred_by_section[section_name] = preds
                    all_label_by_section[section_name] = labels
                    all_path_by_section[section_name] = paths

                    # (Debug) 各データの確認
                    if params.args.debug:
                        print(preds)
                        print(labels)
                        print(paths)
                else:  # 2個目のバッチ以降は出力とラベルを結合していく
                    all_pred_by_section[section_name] = torch.cat((all_pred_by_section[section_name], preds), 0)
                    all_label_by_section[section_name] = torch.cat((all_label_by_section[section_name], labels), 0)
                    all_path_by_section[section_name] = all_path_by_section[section_name] + paths

    return all_pred_by_section, all_label_by_section, all_path_by_section

# 最終全結合層の出力を合わせて平均をとるアンサンブル方法
def ensemble_by_average(params, predicts):
    # モデルからの出力結果のサイズ分のTensorを用意してゼロ初期化する
    pred_averaged_over_all_sections = torch.zeros(predicts['text'].size())
    # 元々のpredと計算デバイスを合わせるために，明示的にデバイスを指定する
    pred_averaged_over_all_sections = pred_averaged_over_all_sections.to(params.device)

    # 3つのモデルの出力結果を足し合わせる
    for section_name in params.yaml['ensemble_section_list']:
        pred_averaged_over_all_sections += predicts[section_name]

    # 足し合わせた出力結果をセクション数で平均する
    pred_averaged_over_all_sections /= len(params.yaml['ensemble_section_list'])

    # 3つの最終全結合層の出力を平均した値の中で最大値となるデータのインデックスを取得する（これが予測クラスになる）
    _, prediction_class = torch.max(pred_averaged_over_all_sections.data, 1)
    return prediction_class


# 最終全結合層の出力を多数決するアンサンブル方法
# TODO: 今後余裕があれば実装する
def ensemble_by_voting(predicts):
    prediction_class = 0
    return prediction_class


def create_bond_vectors(params, features):
    # 3つのモデルから抽出した特徴量を結合する
    for idx, section_name in enumerate(params.yaml['ensemble_section_list']):
        if idx == 0:
            bond_vectors = features[section_name]
        else:
            bond_vectors = torch.cat((bond_vectors, features[section_name]), 1)
    return bond_vectors


# 最終全結合層の出力を学習済みの機械学習アルゴリズムモデルを利用して分類するアンサンブル方法
def ensemble_by_MLalgorithm(params, classifier, features):
    # 3つのモデルから抽出した特徴量を結合する
    bond_vectors = create_bond_vectors(params, features)

    # CPUで計算するようにデータ型を変更しておく
    vector_cpu = bond_vectors.cpu()

    # 学習済みの機械学習アルゴリズムモデルを使ってテストデータに対する予測を行う
    prediction_class = classifier.predict(vector_cpu)

    # この後の精度を計算する関数のためにTensor型に変換した上で返す
    return torch.Tensor(prediction_class).to(params.device)

# アンサンブルとして使用する機械学習アルゴリズムモデルを学習する
def training_MLalgorithm(params, feature_model, data_loaders):
    # 学習済みのDLモデルから学習用のデータにおける特徴量を抽出する
    train_features, train_labels, _ = feature_extraction_by_DLmodel(params, feature_model, data_loaders, data_mode='train')
    val_features, val_labels, _ = feature_extraction_by_DLmodel(params, feature_model, data_loaders, data_mode='val')

    # 事前に決められたモードに従って使用する分類器を決定する

    # モデルから抽出した特徴量を結合する
    train_bond_vectors = create_bond_vectors(params, train_features)
    val_bond_vectors = create_bond_vectors(params, val_features)

    print(train_bond_vectors.shape)

    # CPUで計算するようにデータ型を変更しておく
    train_vectors_cpu = train_bond_vectors.cpu()
    train_labels_cpu = train_labels['text'].cpu()
    val_vectors_cpu = val_bond_vectors.cpu()
    val_labels_cpu = val_labels['text'].cpu()

    features = {'train': train_vectors_cpu, 'val': val_vectors_cpu}
    labels = {'train': train_labels_cpu, 'val': val_labels_cpu}

    # optunaのログをファイルに出力するための設定
    classifier_lists = ['RF', 'SVM', 'LR']
    os.makedirs(os.path.join('./logs/ensemble_model_experiment/optuna'), exist_ok=True)
    optuna_logs_filepath = f'./logs/ensemble_model_experiment/optuna/{classifier_lists[params.args.model_mode - 3]}.log'
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(optuna_logs_filepath))

    # optunaによるハイパラチューニング
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective_variable(params, features, labels), n_trials=100, gc_after_trial=True)

    print(f'Best Train Accuracy: {study.best_value}')
    print(f'Best Parameter: {study.best_params}')

    # 事前に決められたモードに従って使用する分類器を決定する
    if params.args.model_mode == 3:  # Random Forest
        classifier = RandomForestClassifier(n_estimators=study.best_params['n_estimators'],
                                            max_features=study.best_params['max_features'],
                                            max_depth=study.best_params['max_depth'])
    elif params.args.model_mode == 4:  # SVM
        classifier = SVC(gamma=study.best_params['gamma'], C=study.best_params['C'],
                         kernel=study.best_params['kernel'])
    elif params.args.model_mode == 5:  # LF
        classifier = LogisticRegression(max_iter=study.best_params['max_iter'])

    # MLアルゴリズムの分類器を学習する
    classifier.fit(features['train'], labels['train'])

    # 訓練後のvalデータでの精度（accuracy）を出力する
    pred = classifier.predict(features['val'])
    accuracy = accuracy_score(labels['val'], pred)
    print(f'Val Accuracy: {accuracy}')

    return classifier
