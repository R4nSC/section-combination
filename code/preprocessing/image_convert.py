import os
import shutil
import numpy as np
import argparse
from tqdm import tqdm

from .section_images import get_section_name, convert_and_save, make_masked_image, make_deleted_image, make_single_image

# マルウェアを画像に変換する関数(全セクション画像と各セクションをマスクした画像の生成)
def image_convert(params: argparse.Namespace, mode: str):
    # ファミリごとに画像生成
    for label in params.yaml[params.yaml['use_dataset']]['family']:
        # print('Family label ' + label)
        # エラーを吐くデータを避けるディレクトリの作成
        os.makedirs(os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['errors'], label), exist_ok=True)

        # 実験データフォルダ内のファイル名を全て取得
        original_data_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['original'], label)
        files = os.listdir(original_data_path)

        # ファイル１つ１つを取り出す
        with tqdm(files) as pbar:
            pbar.set_description(f'Converting {label:.20}')
            for name in pbar:
                # 実験データフォルダ内の.bytesファイルのみを使用
                if '.bytes' != name[-6:]:
                    continue
                name = name[:-6]
                pbar.set_postfix(progress=name)
                # print('Processing ' + name)

                # .asmファイルから含まれるセクション名をすべて取得する
                section_name = get_section_name(os.path.join(original_data_path, name + '.asm'))

                # HEADERは不必要なので削除
                # (HEADERが存在する保証はされていないので例外処理をプラス)
                try:
                    section_name.remove('HEADER')
                except ValueError:
                    pass

                with open(os.path.join(original_data_path, name + '.bytes')) as file:
                    malware_bytes = []
                    first = -1
                    # for文で回すと1行ずつ文字列として取得できる
                    for line in file:
                        # 文字列を空白文字で分割する
                        data = line.split()

                        # .bytesファイルの最初のアドレスを取得
                        if first == -1:
                            first = int(data[0], 16)

                        # 1行に番地(1つ)と16進数データ(16つ)が含まれている場合のみ使用
                        # (おそらくBIG2015の.bytesファイルがこの形式になっているから)
                        if len(data) != 17:
                            continue

                        # 16進数データを10進数に変換している
                        malware_bytes.append([int(i, 16) if i != '??' else 0 for i in data[1:]])

                    # .bytesファイルの最初のアドレスを取得
                    last = int(data[0], 16) + 16

                if mode == 'masked':
                    for sname in section_name:
                        make_masked_image(params, malware_bytes, name, sname, label, first, last)
                elif mode == 'deleted':
                    # 例外処理 - セクションが1つだけだったら
                    if len(section_name) == 1:
                        # (Debug用) 削除できなかったマルウェアをログに記録する
                        log_path = os.path.join(params.yaml['log']['root'], params.yaml['log']['error'])
                        with open(log_path, 'a', encoding='UTF-8') as error_file:
                            error_file.write('not deleted: {} {}\n'.format(label, name))
                    else:
                        for sname in section_name:
                            make_deleted_image(params, malware_bytes, name, sname, label, first, last)
                elif mode == 'single':
                    for sname in section_name:
                        make_single_image(params, malware_bytes, name, sname, label, first, last)

                # 元画像(加工前)を保存
                save_dir = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs'][mode], 'allSection', label)
                convert_and_save(np.array(malware_bytes), name, save_dir)
                del malware_bytes  # 削除

                # 画像化が終了したマルウェアは作業済みディレクトリに移動
                move_dir = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['converted'], label)
                os.makedirs(move_dir, exist_ok=True)
                shutil.move(os.path.join(original_data_path, name + '.bytes'), move_dir)
                shutil.move(os.path.join(original_data_path, name + '.asm'), move_dir)
