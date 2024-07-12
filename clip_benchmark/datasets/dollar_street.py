import os.path
from pathlib import Path
from typing import Union, Any

import ast
from torchvision.datasets import ImageNet


class DollarStreet(ImageNet):
    def __init__(self, root: Union[str, Path], split: str = "train", **kwargs: Any) -> None:
        imagenet_root = '/mnt/shared/ImageNet-1K-origin/'
        super().__init__(root=imagenet_root, split=split, **kwargs)
        # 更改 imgs samples tragets
        label_root = os.path.join(root, 'images_v2_imagenet_test.csv')

        imgs = []
        targets = []
        # 读取csv文件
        import pandas as pd
        df = pd.read_csv(label_root)
        # 获取 imageRelPath 列和 imagenet_sysnet_id 列
        img_rel_paths = df['imageRelPath']
        imagenet_sysnet_ids = df['imagenet_sysnet_id']
        for img_rel_path, imagenet_sysnet_id in zip(img_rel_paths, imagenet_sysnet_ids):
            if img_rel_path.lower().endswith(self.extensions):
                img_path = os.path.join(root, img_rel_path)
                for id in ast.literal_eval(imagenet_sysnet_id):
                    imgs.append((img_path, id))
                    targets.append(id)
        self.imgs = imgs
        self.samples = imgs
        self.targets = targets
        self.filter_classes()

    def filter_classes(self):
        # Step 1: 筛选出唯一的类别 ID 并升序排列
        unique_targets = sorted(list(set(self.targets)))

        # Step 2: 创建一个类别 ID 到新映射值的字典
        id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(unique_targets)}

        # Step 3: 根据新映射值筛选原始类别列表中的对应名称
        self.targets = [id_to_new_id[old_id] for old_id in self.targets]
        self.classes = [self.classes[old_id] for old_id in unique_targets]

        # 更新 imgs 和 samples
        self.imgs = [(img, id_to_new_id[old_id]) for img, old_id in self.imgs]
        self.samples = self.imgs
