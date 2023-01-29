import os
from tqdm import tqdm
import argparse

from utils import ConfigParameter, make_logger

def malimg_data_shaping(params: argparse.Namespace):
    # ファミリごとに画像生成
    for label in params.yaml['Malimg']['family']:
        # 実験データフォルダ内のファイル名を全て取得
        original_data_path = os.path.join(params.yaml['Malimg']['root'], params.yaml['dirs']['original'], label)
        files = os.listdir(original_data_path)

        with tqdm(files) as pbar:
            pbar.set_description(f'Shaping {label:.20}')
            for name in pbar:
                # 実験データフォルダ内の.asmファイルのみを使用
                if '.asm' != name[-4:]:
                    continue
                name = name[:-4]

                # まだ整形されていないサンプルであれば以降の処理を行う
                # bytes_path = os.path.join(original_data_path, name + '.bytes')
                # if os.path.exists(bytes_path):
                #     continue

                pbar.set_postfix(progress=name)
                with open(os.path.join(original_data_path, name + '.asm')) as file:
                    malware_bytes = []
                    top_address = -1
                    # for文で回すと1行ずつ文字列として取得できる
                    for line in file:
                        # 文字列を空白文字で分割→セクション名とアドレスを':'で分割
                        data = line.split()
                        _section_name, now_address = data[0].split(':')

                        # .bytesファイルの最初のアドレスを取得
                        if top_address == -1:
                            top_address = int(now_address, 16)
                        else:
                            # データが欠損している場合は'00'で補完する
                            if int(now_address, 16) - before_address > before_byte_num:
                                # 欠損しているバイト数を計算して，そのバイト数分補完する
                                add_byte_num = (int(now_address, 16) - before_address) - before_byte_num
                                malware_bytes.extend(['00'] * add_byte_num)

                        # バイト列をリストに追加する
                        malware_bytes.extend(data[1:])
                        before_address = int(now_address, 16)
                        before_byte_num = len(data) - 1

                # アドレスおよび16バイト分のデータが各行に出力された.bytesファイルを作成
                with open(os.path.join(original_data_path, name + '.bytes'), 'w') as output_file:
                    output_index = 0
                    # バイト列が残っている間，処理を続ける
                    while len(malware_bytes) > output_index:
                        output_data = [str(i) for i in malware_bytes[output_index:output_index+16]]
                        output_string = f'{"%08x" % (top_address + output_index)} {" ".join(output_data)}\n'
                        output_file.write(output_string)
                        output_index += 16


if __name__ == '__main__':
    params = ConfigParameter()
    logger = make_logger()

    malimg_data_shaping(params)
