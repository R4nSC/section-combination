# section-combination

[SCIS2023で発表した研究](https://sec.inf.shizuoka.ac.jp/publications/20230125_TMNO2023/#:~:text=Jan%202023.%0A%20%5B-,Paper,-%5D)に使用したプロジェクトである．PEファイルの各セクションのみを抽出した単体画像を作成し，それぞれの画像で学習したモデルを複数組み合わせるアンサンブル型マルウェア分類器の提案である．

## ディレクトリ構成

> **Warning**
> 重要な箇所のみを説明しており，一部省略しています

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
 │  │  ├───── parse_args.py  # コマンドライン引数およびyamlからのパラメータ読み込み
 │  │  ├───── logger.py  # ロガー関係
 │  │  └───── optuna_objective.py  # Optunaの評価関数
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

> **Note**
> 実験に利用した検証環境です．SCIS2023に投稿した論文内に記載されたものと同様です．

- ハードウェア
    - AMD Ryzen9 3950X プロセッサ
    - 64GB メモリ
    - GeForce RTX 2080 Ti (11GB GDDRメモリ)
- ソフトウェア
    - Ubuntu 20.04.1 LTS
    - Linux 5.4.0-56-generic
    - Python 3.6.10
    - PyTorch 1.8.0
    - scikit-learn 0.23.2

## インストールおよび実行方法

1. このプロジェクトをcloneする，もしくはZipファイルをダウンロードして解凍する．

2. データセットのダウンロードを行う．
    1. BIG2015の場合
        - [kaggle](https://www.kaggle.com/competitions/malware-classification/data)の`train.7z`および`trainLabels.csv`をダウンロードおよび解凍し，`resources/BIG2015`にそれぞれのファイルを保存する
        - `resources/BIG2015/train`内に各サンプルをファミリごとにディレクトリを分けて保存する
        > **Note**
        > サンプルをファミリごとに分けるコードが見つかりませんでした...
        > 自分で実装をお願いします...
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
    > **Warning**
    > parse_args.pyに下記以上のパラメータが指定できるよう書いてあるが，途中でプログラムの全面改変を行なっているため，動作しない引数がある可能性あり．
    > 動作を確認しているパラメータを下記に記載していることを留意してほしい．

    - コマンドライン引数によって検証パラメータを指定する
        - `-n, --network` : 利用するCNNモデル（"vgg16", "resnet50"）
        - `-e, --epochs` : エポック数
        - `-b, --batch-size` : バッチサイズ数
        - `-m, --model-mode` : アンサンブルモデルの種類（0->original, 1->average, 3->RF, 4->SVM, 5->LR, 6->voting）
        - `-d, --dataset` : データセット（"BIG2015", "Malimg"）

## Note

プログラムについて分からないことがあれば@takeuchiまで

## License

本プロジェクトは社外秘（Confidential）である．
