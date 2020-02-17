class Config:
    def __init__(self):
        self.batch_size = 64
        self.lr = 1e-3
        self.label_path = "/data/Heart-rate/train/cropped/valid-labels.txt"
        self.split_rate = 0.3
        self.num_classes = 128
        self.losses = ['mse']
        self.num_epoch = 2000
        self.val_freq = 10
        self.milestones = [800, 1500]
        self.num_workers = 8
        self.clip_len = 64
        self.k_fold = 5
        self.num_samples = 5