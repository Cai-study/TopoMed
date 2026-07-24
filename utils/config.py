class Config:
    def __init__(self,task):
        if "mmwhs" in task:
            self.save_dir = './data'
            self.patch_size =(128, 128, 128)
            self.num_cls = 5
            self.num_channels = 1
            self.n_filters = 32
        else:
            raise NameError("Please provide correct task name, see config.py")


