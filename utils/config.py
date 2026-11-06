class Config:
    def __init__(self, task, save_dir=None, num_cls=None, patch_size=None, batch_size=2):
        if "mmwhs" in task:
            self.save_dir = './Datasets/MMWHS_data'
            self.patch_size = (128, 128, 128)
            self.num_cls = 5
            self.num_channels = 1
            self.n_filters = 32
        else:
            if save_dir is None or num_cls is None or patch_size is None:
                raise ValueError('For non-MMWHS data supply --data_dir, --num_classes and --patch_size D H W')
            self.save_dir, self.num_cls, self.patch_size = save_dir, num_cls, tuple(patch_size)
            self.num_channels, self.n_filters = 1, 32
        self.save_dir = save_dir if save_dir is not None else self.save_dir
        self.num_cls = num_cls if num_cls is not None else self.num_cls
        self.patch_size = tuple(patch_size) if patch_size is not None else self.patch_size
        self.batch_size = batch_size


