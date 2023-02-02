import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# optunaの目的関数を設定する
def objective_variable(params: argparse.Namespace, features: dict, labels: dict):
    """optunaの目的関数を設定する

    Parameters
    ----------
    params : argparse.Namespace
        読み込んだパラメータ
    features : dict
        学習および検証に利用する特徴量
    labels : dict
        学習および検証に利用する特徴量

    Returns
    -------
    function
        パラメータに応じた目的関数
    """
    def objective(trial):
        # optunaでチューニングするパラメータを設定する
        # model_modeによって使用する分類器を切り替える
        classifier = None
        if params.args.model_mode == 3:  # LF
            # criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            # bootstrap = trial.suggest_categorical('bootstrap', ['True', 'False'])
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            max_depth = trial.suggest_int('max_depth', 1, 500)
            # max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 1000)
            n_estimators = trial.suggest_int('n_estimators', 1, 300)
            # min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            # min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            classifier = RandomForestClassifier(n_estimators=n_estimators,
                                                max_features=max_features, max_depth=max_depth)
        elif params.args.model_mode == 4:  # SVM
            C_num = trial.suggest_uniform('C', 0.001, 100)
            gamma = trial.suggest_uniform('gamma', 0.0001, 0.1)
            kernel = trial.suggest_categorical('kernel', ['poly', 'rbf'])
            classifier = SVC(C=C_num, gamma=gamma, kernel=kernel)
        elif params.args.model_mode == 5:  # LR
            # solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag'])
            # C_num = trial.suggest_loguniform('C', 0.001, 100)
            max_iter = trial.suggest_int('max_iter', 100, 100000)
            classifier = LogisticRegression(max_iter=max_iter)

        # trainで訓練した後に，valを予測する
        classifier.fit(features['train'], labels['train'])
        pred = classifier.predict(features['val'])

        # accuracyのスコアで最適化する
        accuracy = accuracy_score(labels['val'], pred)
        return accuracy

    return objective
