from torch.utils.data import Dataset

class RPPG_Dataset(Dataset):
    def __init__(self, data_path = "", mode = "train"):
        self.mode = mode
    
    def _load_data(self):
        pass

    def __len__(self):
        pass
    
    def __getitem__(self,index):
        pass
