from .evaluation_phase import evaluation_of_ensemble_model
from .training_phase import train_of_ensemble_model

# アンサンブル型マルウェア分類モデルの学習および評価
def train_and_evaluation_of_ensemble_model(params, logger, data_loaders, data_sizes):
    # 2. 3セクションの単体画像で各モデルを学習する
    model_ft = {}

    # 元画像でもモデルを学習および評価する（比較実験）
    if params.args.model_mode == -1:
        logger.info("Started training model by allSection images.")
        section_name = 'allSection'
        model_ft[section_name], accuracy = train_of_ensemble_model(params, data_loaders[section_name], data_sizes[section_name], section=section_name)
        logger.info("Finished training model by allSection images.")

    else:
        # 各セクション単体画像のモデルを学習および評価する
        logger.info("Started training model by section stand-alone images.")
        for section_name in params.yaml['load_section_list']:
            # # モデルを訓練するもしくは訓練済みのパラメータをそのまま使用する
            # model_filename = f'{section_name}_{args.network}_e{args.epochs}_b{args.batch_size}_ver{MODEL_FILE_VERSION}.pth'
            # model_path = f'{MODEL_DIR}/{model_filename}'

            logger.info("Started training model by %s images.", section_name)
            print(f"--- Train {section_name} image model ---\n")
            model_ft[section_name], _result = train_of_ensemble_model(params, data_loaders[section_name], data_sizes[section_name], section=section_name)
            if params.args.model_use:
                logger.info("Used model previously trained by %s images.", section_name)
            logger.info("Finished training model by %s images.", section_name)

            # # モデルのパラメータを保存する
            # if args.ensemble_model_save:
            #     torch.save(model_ft[section_name].model.network.state_dict(), model_path)
            #     logger.info("Saved model trained by %s images.", section_name)

        logger.info("Finished training model by section stand-alone images.")

        # 3モデルの結果をアンサンブルして最終的な結果を出す
        accuracy = {}
        logger.info("Started evaluating ensemble model.")
        # accuracy = evaluation_of_ensemble_model(params, logger, model_ft, data_loaders)
        # 組み込みなし
        for mode in params.yaml['all_model_mode']:
            print(f"--- Test ensemble model Number {mode}---\n")
            params.args.model_mode = mode
            accuracy[mode] = evaluation_of_ensemble_model(params, logger, model_ft, data_loaders)

        # 組み込みあり
        params.yaml['ensemble_section_list'].insert(0, 'allSection')
        for mode in params.yaml['all_model_mode']:
            print(f"--- Test ensemble model Number builtin-{mode}---\n")
            params.args.model_mode = mode
            accuracy[mode + 10] = evaluation_of_ensemble_model(params, logger, model_ft, data_loaders)
        logger.info("Finished evaluating ensemble model.")

    return accuracy
