import os
import numpy as np
import copy
import argparse
from components import judge_section_boundary, convert_and_save


def single_section_bytes(malware_bytes: list, asm_path: str, section: str, first: int, last: int) -> list:
    """指定セクション単体のマルウェアのバイト列を返す

    Parameters
    ----------
    malware_bytes : list
        マルウェアのバイト列
    asm_path : str
        対象.asmファイルのパス
    section : str
        セクション名
    first : int
        対象ファイルの開始アドレス
    last : int
        対象ファイルの終了アドレス

    Returns
    -------
    list
        対象セクションのみのバイト列
    """
    # 指定セクションの開始地点および終了地点を取得
    section_start_address, section_end_address = judge_section_boundary(asm_path, section, last)
    # 作業用配列をコピー
    single_bytes = copy.copy(malware_bytes)

    now_address = first
    not_delete_index = []
    # 取り除かないバイト列のインデックスを保持する
    for index in range(np.array(single_bytes).shape[0]):
        for j in range(len(section_start_address)):
            # 1. 現在見ているアドレスが対象セクションである場合 -> 削除するインデックスを保持
            if now_address >= section_start_address[j] and now_address + 15 < section_end_address[j]:
                not_delete_index.append(index)
            # 2. 現在アドレスと現在アドレス+16の間に開始アドレスがある場合 -> 開始アドレスよりも小さい方だけ0に変える(妥協)
            elif now_address < section_start_address[j] <= now_address + 15:
                single_bytes[index] = [malware_bytes[index][i] if now_address + i >= section_start_address[j] else 0 for i in range(16)]
                not_delete_index.append(index)
            # 3. 現在アドレスと現在アドレス+16の間に終了アドレスがある場合 -> 終了アドレスよりも大きい方だけ0に変える(妥協)
            elif now_address < section_end_address[j] <= now_address + 15:
                single_bytes[index] = [malware_bytes[index][i] if now_address + i < section_end_address[j] else 0 for i in range(16)]
                not_delete_index.append(index)
        now_address += 16

    # 削除しないバイト列(該当セクション)だけのリストを作る
    new_bytes = []
    for index in range(len(not_delete_index)):
        new_bytes.append(single_bytes[not_delete_index[index]])

    return new_bytes


def make_single_image(params: argparse.Namespace, malware_bytes: list, filename: str, section: str, label: str, first: int, last: int) -> None:
    """対象セクション単体の画像を保存する

    Parameters
    ----------
    params : argparse.Namespace
        読み込んだパラメータ
    malware_bytes : list
        マルウェアのバイト列
    filename : str
        サンプル名
    section : str
        セクション名
    label : str
        ファミリー名
    first : int
        対象ファイルの開始アドレス
    last : int
        対象ファイルの終了アドレス
    """
    asm_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['original'], label, filename, '.asm')  # .asm用ファイルパス

    # 対象セクションのみのバイナリ配列を作成する
    single_section_bytes = single_section_bytes(malware_bytes, asm_path, section, first, last)

    # 例外処理 - 何もないデータだったら
    if np.array(single_section_bytes).shape[0] == 0:
        # (Debug用) 作成できなかったマルウェアをログに記録する
        log_path = os.path.join(params.yaml['log']['root'], params.yaml['log']['error'])
        with open(log_path, 'a', encoding='UTF-8') as error_file:
            error_file.write('not created single images: {} {}\n'.format(label, filename))
    else:
        # 指定されたパスに画像を保存する
        dir_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['single'], (section[1:] if section[0] == '.' else section), label)
        convert_and_save(np.array(single_section_bytes), filename, dir_path)
        del single_section_bytes  # 一応削除
