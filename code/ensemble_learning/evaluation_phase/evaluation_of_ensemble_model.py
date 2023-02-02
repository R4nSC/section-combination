from .feature_extraction_and_ensemble import feature_extraction_by_DLmodel, training_MLalgorithm, ensemble_by_average, ensemble_by_voting, ensemble_by_MLalgorithm
from .classification_test import calc_results_from_predicts_and_labels

# アンサンブル型モデル用の分類タスク評価
def evaluation_of_ensemble_model(params, logger, features_models, data_loaders):
    # 3つのモデルの最終全結合層の出力とラベルを持ってくる
    all_pred_by_section, all_label_by_section, all_path_by_section = feature_extraction_by_DLmodel(params,
                                  features_models,
                                  data_loaders,
                                  data_mode='test')
    print(all_pred_by_section['text'].shape)

    # 指定されたモード番号に応じて各学習モデルをアンサンブルする方法を切り替える
    if params.args.model_mode == 1:  # 平均
        prediction_class = ensemble_by_average(params, all_pred_by_section)
    elif params.args.model_mode == 2:  # 多数決 TODO:今後実装予定
        prediction_class = ensemble_by_voting(all_pred_by_section)
    else:  # MLアルゴリズムを使った分類
        logger.info("Started learning ensemble with machine learning algorithm.")
        ML_classifier = training_MLalgorithm(params, features_models, data_loaders)
        logger.info("Finished learning ensemble with machine learning algorithm.")

        prediction_class = ensemble_by_MLalgorithm(params, ML_classifier, all_pred_by_section)

    # 分類精度を算出して標準出力する
    accuracy = calc_results_from_predicts_and_labels(params, prediction_class,
                                                     all_label_by_section['text'], all_path_by_section['text'])
    return accuracy
