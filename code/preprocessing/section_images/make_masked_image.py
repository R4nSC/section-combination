import os
import numpy as np
import copy
import argparse
from components import judge_section_boundary, convert_and_save


def masked_section_bytes(malware_bytes: list, asm_path: str, section: str, first: int, last: int) -> list:
    """指定セクションをマスクしたマルウェアのバイト列を返す

    マスク: 0(黒)にする

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
        対象セクションをマスクしたバイト列
    """
    # 指定セクションの開始地点および終了地点を取得
    section_start_address, section_end_address = judge_section_boundary(asm_path, section, last)
    # 作業用配列をコピー
    masked_bytes = copy.copy(malware_bytes)

    now_address = first
    # 開始地点と終了地点の間のバイトを0に変更する
    for index in range(np.array(masked_bytes).shape[0]):
        for j in range(len(section_start_address)):
            # 1. 現在見ているアドレスが対象セクションである場合 -> すべてのバイトを0に変える
            if now_address >= section_start_address[j] and now_address + 15 < section_end_address[j]:
                masked_bytes[index] = [0 for i in range(16)]
            # 2. 現在アドレスと現在アドレス+16の間に開始アドレスがある場合 -> 開始アドレスよりも小さい方だけ0に変える
            elif now_address < section_start_address[j] <= now_address + 15:
                masked_bytes[index] = [malware_bytes[index][i] if now_address + i < section_start_address[j] else 0 for i in range(16)]
            # 3. 現在アドレスと現在アドレス+16の間に終了アドレスがある場合 -> 終了アドレスよりも大きい方だけ0に変える
            elif now_address < section_end_address[j] <= now_address + 15:
                masked_bytes[index] = [malware_bytes[index][i] if now_address + i >= section_end_address[j] else 0 for i in range(16)]
        now_address += 16
    return masked_bytes


def make_masked_image(params: argparse.Namespace, malware_bytes: list, filename: str, section: str, label: str, first: int, last: int) -> None:
    """対象セクションをマスクした画像を保存する

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

    # 対象セクションをマスクしたバイナリ配列を作成する
    masked_section_bytes = masked_section_bytes(malware_bytes, asm_path, section, first, last)

    # 指定されたパスに画像を保存する
    dir_path = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['masked'], (section[1:] if section[0] == '.' else section), label)
    convert_and_save(np.array(masked_section_bytes), filename, dir_path)
    del masked_section_bytes  # 一応削除
