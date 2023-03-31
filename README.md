# section-combination

## 概要

[SCIS2023で発表した研究](https://sec.inf.shizuoka.ac.jp/publications/20230125_TMNO2023/#:~:text=Jan%202023.%0A%20%5B-,Paper,-%5D)に使用したプロジェクトである．PEファイルの各セクションのみを抽出した単体画像を作成し，それぞれの画像で学習したモデルを複数組み合わせるアンサンブル型マルウェア分類器の提案である．

## ディレクトリ構成

```shell-session
section-combination/
 ├───── README.md
 ├───── .gitignore
 ├──┬── code/  # 基本的にコードを書くディレクトリ
 │  ├──┬── datasets  # データセット関係
 │  │  └───── load_datasets.py  # データセットの読み込み
 │  ├───── ensemble_learning  # アンサンブル学習関係
 │  ├───── models  # モデル関係
 │  ├──┬── preprocessing  # 前処理関係
 │  │  ├──┬── section_images  # セクションに特定の追加処理を施す画像生成
 │  │  │  ├───── make_masked_image.py  # マスク画像生成
 │  │  │  ├───── make_deleted_image.py  # 削除画像生成
 │  │  │  └───── make_single_image.py  # 単体画像生成
 │  │  ├───── image_convert.py  # 画像生成
 │  │  └───── shaping_dataset.py  # ディレクトリの作成
 │  ├───── utils  # その他
 │  ├───── experiment_of_ensemble_model.py  # アンサンブルモデル実験
 │  ├───── experiment_of_integrated_ensemble_model.py  # アンサンブルモデル実験（特徴統合型）
 │  ├───── malimg_setup.py  # Malimg特有の前処理
 │  └───── setup.py  # 画像生成などの前処理
 ├──┬── yaml/
 │  └───── config.yml  # プログラムに読み込むパラメータ
 ├──┬── resources/  # 実験に利用するデータ
 │  ├───── BIG2015/  # BIG2015データセット
 │  └───── Malimg/  # Malimgデータセット
 └───── .vscode/  # 作成者のvscode設定（）
```

## 動作環境

## インストールおよび実行方法

1. このプロジェクトをcloneする，もしくはZipファイルをダウンロードして解凍する．

2. データセットのダウンロードを行う．
    1. BIG2015の場合
        - [kaggle](https://www.kaggle.com/competitions/malware-classification/data)の`train.7z`および`trainLabels.csv`をダウンロードおよび解凍し，`resources/BIG2015`にそれぞれのファイルを保存する
        - `resources/BIG2015/train`内に各サンプルをファミリごとにディレクトリを分けて保存する
        :::note warn
        サンプルをファミリごとに分けるコードが見つかりませんでした...自分で実装をお願いします...
        :::
    2. Malimgの場合
        - [PapersWithCode](https://paperswithcode.com/dataset/malimg)の`malimg_dataset.zip`をダウンロードおよび解凍し，`resources/Malimg`にそれぞれのファイルを保存する

3. `code/setup.py`を実行し，元画像および単体画像の生成，およびアンサンブルモデルの検証に利用するデータを抽出したディレクトリの作成を行う．
    - `yaml/config.yml`内の`use_dataset`に画像を生成したいデータセットを指定する（現在，指定できるのは`BIG2015`／`Malimg`）
    - コマンドライン引数で`-s`を利用して生成する画像を指定する（現在，指定できるのは`masked`／`deleted`／`single`）
    - 画像の生成が途中で中断された場合，途中から実行されるような仕様になっているが，1から画像の生成を行いたい場合は`--setup-reset`を指定する

    ---
    ```shell-session
    $ cd code
    $ python setup.py -s single
    ```

4. `resources/`内の画像を生成したデータセットのディレクトリに`ensemble/`ディレクトリがあることを確認し，`code/experiment_of_ensemble_model.py`もしくは`code/experiment_of_integrated_ensemble_model.py`の実験用プログラムを実行する．
