import os
import shutil
import sys
sys.path.append('../')

from utils import ImageFolderWithPaths
from .section_images import get_section_name

# 単体画像データセットフォルダ(resource/BIG2015/single_images/)から
# アンサンブル学習用のデータセットフォルダ(resource/BIG2015/ensemble)を作成する
# ※.text/.rdata/.dataの3セクションをすべて持つサンプルのみを抽出したフォルダを作る目的
def create_dataset_folder_for_ensemble(params, logger):
    print("--- Create a dataset folder for ensemble-learning ---\n")

    # 保存するディレクトリ(ensemble/)がなければ作成する
    ensemble_data_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['ensemble'])

    # .text/.rdata/.dataの3セクションをすべて持つサンプルのみを抽出する
    single_data_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['single'])

    # 指定セクションの単体画像のImageFolderを作成
    allSection_image_folder = ImageFolderWithPaths(os.path.join(single_data_path, 'allSection'), params.transforms)

    # ImageFolderから画像を1つずつ取り出して.text/.rdata/.dataの3セクションをすべて含んでいるか判定
    for i in range(len(allSection_image_folder)):
        # 画像のパスを取得
        allSection_image_path = allSection_image_folder[i][2]

        # '/'で文字列を区切り，ファイル名とラベル（ファミリー名）を取得
        path_split_by_slash = allSection_image_path.split('/')
        filename = path_split_by_slash[-1]
        label = path_split_by_slash[-2]

        # 指定サンプルに含まれたセクション名をすべて列挙する
        asm_file_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['converted'], label, filename[:-4] + '.asm')
        list_of_section_names_included_sample = get_section_name(asm_file_path)

        # 定数のセクション名リストと比較するために'.'を除去
        for j, name in enumerate(list_of_section_names_included_sample):
            list_of_section_names_included_sample[j] = name.replace(".", "")

        # 対象の3セクションが含まれていたらアンサンブル用のデータセットフォルダに保存
        if set(list_of_section_names_included_sample) >= set(params.yaml['ensemble_section_list']):
            for j, section_name in enumerate(params.yaml['load_section_list']):
                # 画像パスを保存するセクションのものに変更する]
                single_image_path = allSection_image_path
                if j != 0:
                    single_image_path = single_image_path.replace(params.yaml['load_section_list'][j-1], section_name)

                # 保存するディレクトリがなければ作成する
                os.makedirs(os.path.join(ensemble_data_path, section_name, label), exist_ok=True)

                # 対象画像ファイルをコピーする
                shutil.copy(single_image_path, os.path.join(ensemble_data_path, section_name, label, filename))

            original_image_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['images'], label, filename[:-4] + '.png')
            # 保存するディレクトリがなければ作成する
            os.makedirs(os.path.join(ensemble_data_path, 'allSection_original', label), exist_ok=True)
            # 対象画像ファイルをコピーする
            shutil.copy(original_image_path, os.path.join(ensemble_data_path, 'allSection_original', label, filename))
    print("--- Finished creating a dataset folder for ensemble-learning ---\n")
