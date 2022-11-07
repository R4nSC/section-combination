from torchvision import datasets


class ImageFolderWithPaths(datasets.ImageFolder):
    """画像のファイルパスを追加で保持したImageFolder

    Returns
    -------
    tuple
        通常のImageFolderの戻り値に画像のファイルパスを追加したタプル
    """
    def __getitem__(self, index):
        # 通常のImageFolderの戻り値
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # 画像のファイルパス
        path = self.imgs[index][0]
        # オリジナルの戻り値にファイルパスを追加したタプルを作成する
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
