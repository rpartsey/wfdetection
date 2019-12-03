from __future__ import print_function, division
import random

from .binary_dataloader import BinaryLoader


class BasePositiveLoader(BinaryLoader):
    def __init__(self, csv_file, *args, **kwargs):
        data = csv_file[csv_file[self.MASK_EXISTS_COLUMN]]
        super(BasePositiveLoader, self).__init__(data, *args, **kwargs)

    @classmethod
    def build_from_binary_loader(cls, binary_loader):
        if not isinstance(binary_loader, BinaryLoader):
            raise ValueError('input arg `binary_loader` should be instance of BinaryLoader')
        params = dict(
            csv_file=binary_loader.input_df,
            root_dir=binary_loader.root_dir,
            image_transform=binary_loader.image_transform,
            mask_transform=binary_loader.mask_transform,
            aug_transform=binary_loader.aug,
            cut_borders=binary_loader.cut_borders,
            align=binary_loader.align)
        return cls(**params)


class TrainLoader(BasePositiveLoader):
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.input_df) - 1)
        return super(TrainLoader, self).__getitem__(idx)


class ValLoader(BasePositiveLoader):
    def __len__(self):
        return 10
