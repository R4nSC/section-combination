import sys
import logging


def make_logger() -> logging:
    """ロガーを生成して返す

    Returns
    -------
    logging
        生成したロガーインスタンス
    """

    # ロガーの生成
    logger = logging.getLogger('log')
    # 出力レベルの設定
    logger.setLevel(logging.INFO)
    # ハンドラの生成
    handler = logging.StreamHandler(sys.stderr)
    # ロガーにハンドラを登録
    logger.addHandler(handler)
    # フォーマッタの生成
    fmt = logging.Formatter('[%(asctime)s] %(levelname)7s -> %(message)s (%(filename)s:%(funcName)s)')
    # ハンドラにフォーマッタを登録
    handler.setFormatter(fmt)

    return logger
