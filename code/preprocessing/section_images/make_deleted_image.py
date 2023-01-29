import os
import numpy as np
import copy
import argparse
from .components import judge_section_boundary, convert_and_save


def create_deleted_section_bytes(malware_bytes: list, asm_path: str, section: str, first: int, last: int) -> list:
    """指定セクションを削除したマルウェアのバイト列を返す

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
        対象セクションを削除したバイト列
    """
    # 指定セクションの開始地点および終了地点を取得
    section_start_address, section_end_address = judge_section_boundary(asm_path, section, last)
    # 作業用配列をコピー
    deleted_bytes = copy.copy(malware_bytes)

    now_address = first
    deleted_index = []
    # 取り除くバイト列のインデックスを保持する
    for index in range(np.array(deleted_bytes).shape[0]):
        for j in range(len(section_start_address)):
            # 1. 現在見ているアドレスが対象セクションである場合 -> 削除するインデックスを保持
            if now_address >= section_start_address[j] and now_address + 15 < section_end_address[j]:
                deleted_index.append(index)
            # 2. 現在アドレスと現在アドレス+16の間に開始アドレスがある場合 -> 開始アドレスよりも小さい方だけ0に変える(妥協)
            elif now_address < section_start_address[j] <= now_address + 15:
                deleted_bytes[index] = [malware_bytes[index][i] if now_address + i < section_start_address[j] else 0 for i in range(16)]
            # 3. 現在アドレスと現在アドレス+16の間に終了アドレスがある場合 -> 終了アドレスよりも大きい方だけ0に変える(妥協)
            elif now_address < section_end_address[j] <= now_address + 15:
                deleted_bytes[index] = [malware_bytes[index][i] if now_address + i >= section_end_address[j] else 0 for i in range(16)]
        now_address += 16

    # 実際にバイト列を取り除く
    for index in range(len(deleted_index)):
        del deleted_bytes[deleted_index[len(deleted_index) - index - 1]]

    return deleted_bytes


def make_deleted_image(params: argparse.Namespace, malware_bytes: list, filename: str, section: str, label: str, first: int, last: int) -> None:
    """対象セクションを削除した画像を保存する

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
    asm_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['original'], label, filename + '.asm')  # .asm用ファイルパス

    # 対象セクションを削除したバイナリ配列を作成する
    deleted_section_bytes = create_deleted_section_bytes(malware_bytes, asm_path, section, first, last)

    # 例外処理 - 削除したときにデータがすべてなくなったら
    if np.array(deleted_section_bytes).shape[0] == 0:
        # (Debug用) 削除できなかったマルウェアをログに記録する
        log_path = os.path.join(params.yaml['log']['root'], params.yaml['log']['error'])
        with open(log_path, 'a', encoding='UTF-8') as error_file:
            error_file.write('not deleted: {} {}\n'.format(label, filename))
    else:
        # 指定されたパスに画像を保存する
        dir_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['deleted'], (section[1:] if section[0] == '.' else section), label)
        convert_and_save(np.array(deleted_section_bytes), filename, dir_path)
        del deleted_section_bytes  # 一応削除
