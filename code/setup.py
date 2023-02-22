import os
import shutil
import argparse

from preprocessing import image_convert, create_dataset_folder_for_ensemble
from utils import ConfigParameter, make_logger

def reset_setup(params: argparse.Namespace, before_dir: str, after_dir: str):
    for label in params.yaml[params.yaml['use_dataset']]['family']:
        before_path = os.path.join(before_dir, label)
        if os.path.exists(before_path) == False:
            continue
        after_path = os.path.join(after_dir, label)
        os.makedirs(after_path, exist_ok=True)
        files = os.listdir(before_path)
        os.makedirs(after_path, exist_ok=True)

        for name in files:
            shutil.move(os.path.join(before_path, name), after_path)

if __name__ == '__main__':
    params = ConfigParameter()
    logger = make_logger()

    # # リセットフラグが立っていたらデータ(converted/)を元のディレクトリ(train/)に戻す
    # if params.args.setup_reset:
    #     converted_dir = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['converted'])
    #     original_dir = os.path.join(params.yaml[params.yaml['use_dataset']]['root'], params.yaml['dirs']['original'])
    #     reset_setup(params, converted_dir, original_dir)

    # logger.info(f"Started converted malware binaries to {params.args.setup_mode} images.")
    # image_convert(params, mode=params.args.setup_mode)
    # logger.info(f"Finished converted malware binaries to {params.args.setup_mode}  images.")

    if params.args.setup_mode == "single":
        create_dataset_folder_for_ensemble(params, logger)
