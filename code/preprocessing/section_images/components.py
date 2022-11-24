import os
import argparse
import numpy as np
from typing import Tuple
from math import log
from PIL import Image

def get_section_name(asm_path: str) -> list:
    """.asmファイルから含まれるセクション名のリストを取得する

    Parameters
    ----------
    asm_path : str
        対象.asmファイルのパス

    Returns
    -------
    list
        重複を無くしたセクション名のリスト
    """
    sectionName = []  # セクションラベル配列

    # .asmファイルを開いて':'以前の文字列をリストに追加する
    # 例: ".text:00401000 ... " -> ".text"を抽出
    with open(asm_path, encoding="ISO-8859-1") as file:
        for line in file:
            data = line.split(':')
            sectionName.append(data[0])
    return list(set(sectionName))


def judge_section_boundary(asm_path: str, section: str, last: int) -> Tuple[list, list]:
    """.asmファイルから指定セクションの開始点と終了点を取得する

    Parameters
    ----------
    asm_path : str
        対象.asmファイルのパス
    section : str
        セクション名
    last : int
        マルウェアサンプルの最終アドレス

    Returns
    -------
    Tuple[list, list]
        開始アドレスリスト，終了アドレスリスト
    """
    start = []  # セクションの開始点リスト
    end = []  # セクションの終了点リスト

    with open(asm_path, encoding="ISO-8859-1") as file:
        for line in file:
            data = line.split(':')
            if len(data) > 1:
                # 開始点発見なし && 指定セクションであった場合(セクションの開始点)
                if len(start) == len(end):
                    if data[0] == section:
                        start.append(int(data[1][:8], 16))
                # 開始点発見済み && 指定セクションでない場合(セクションの終了点)
                else:
                    if data[0] != section:
                        end.append(int(data[1][:8], 16))
        # 最終セクションの終了アドレスを追加
        if len(start) != len(end):
            end.append(last)
        return start, end


# バイナリを画像に変換し，その画像を保存する関数
def convert_and_save(binary: np.array, filename: str, dir_path: str) -> Image:
    """バイナリを画像に変換し，その画像を保存する

    Natarajらの論文で指定されている画像サイズに整えた後に256×256にリサイズしている

    Parameters
    ----------
    binary : np.array
        画像に変換するバイナリが格納された配列
    filename : str
        サンプル名
    dir_path : str
        画像の保存先ディレクトリ

    Returns
    -------
    Image
        バイナリから変換した画像インスタンス
    """
    # 2次元配列(binary)の横幅が16でない場合は例外を出す
    if binary.shape[1] != 16:
        assert False

    # バイナリファイルのサイズに応じて画像サイズを決める
    b = int((binary.shape[0] * 16) ** 0.5)
    b = 2 ** (int(log(b) / log(2)) + 1)
    a = int(binary.shape[0] * 16 / b)
    binary = binary[:a * b // 16, :]

    # 決めた画像サイズに整える
    binary = np.reshape(binary, (a, b))

    # 画像に変換する
    im = Image.fromarray(np.uint8(binary))

    # 画像を256×256にリサイズする
    im = im.resize((256, 256))

    # 保存するディレクトリがなければ作成する
    os.makedirs(dir_path, exist_ok=True)

    # 指定したパスに変換した画像を保存する
    im.save(os.path.join(dir_path, filename + '.png'), "PNG")
    return im
