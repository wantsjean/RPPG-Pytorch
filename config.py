class Cfg:
    def __init__(self):
        self.batch_size = 32
        self.lr = 1e-3
        self.data_root = ""
        self.split_rate = 0.3
        self.num_classes = 128
        self.losses = ['ce','mse']
        self.num_epoch = 500
        self.val_freq = 10
        self.milestones = [300, 400]
        self.num_workers = 4