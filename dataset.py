from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import random
import os
# import tkinter
# import matplotlib
# matplotlib.use('TkAgg')
# import seaborn as sns
# import matplotlib.pyplot as plt

os.environ['DISPLAY'] = ':13'

class RPPG_Dataset(Dataset):
    
    def __init__(self, label_path, mode = "train", clip_len=64, cur=0, k_fold=5, rng_seed=0):

        assert mode in ['train', 'test', 'val']
        self.mode = mode
        self.clip_len = clip_len
        # load data_path, labels and vid_idx
        self.vid_meta = self._load_data(label_path)
        # count and visualize
        # self._count()
        # if training, split the data by val_rate
        if mode in ['train', 'val']:
            self._split_data(cur, k_fold, rng_seed)

        # if mode in ['val', 'test']:
        #     self._dense_sample()

    def _load_data(self, label_path):
        # load with pandas
        keys =["vid_name","vid_path","data_path","fps","hr"]
        data = pd.read_table(label_path, sep=', ', header=None, names=keys)

        # convert string to float
        data["fps"] = pd.to_numeric(data["fps"],errors='coerce')
        data["hr"] = pd.to_numeric(data["hr"],errors='coerce')

        # drop NaN
        data.dropna(axis=0, how='any', inplace=True)
        data.reset_index(drop=True)
        
        vid_meta = {}
        
        # conver from Dataframe to list
        for key in keys:
            vid_meta[key] = data[key].tolist()

        return vid_meta
    
    def _split_data(self, cur = 0, k_fold=5, rng_seed = 0):
        
        assert 0 <= cur < k_fold
        # set the rng seed
        random.seed(rng_seed)
        # total 500 persons in the training set
        candidates = [[] for _ in range(500)]
        # split the data by mod 5 (every person contains 5 videos)
        for i, vid_name in enumerate(self.vid_meta["vid_name"]):
            vid_idx = int(vid_name.split('-')[1])
            candidates[vid_idx//5].append(i)
        # shuffle for constructing train-val set
        random.shuffle(candidates)
        # choose fold by cur
        if self.mode == 'train':
            candidates = candidates[:int(cur*500/k_fold)]+candidates[int((cur+1)*500/k_fold):]
        else:
            candidates = candidates[int(cur*500/k_fold):int((cur+1)*500/k_fold)]
        # flatten candidates
        candidates = [i for sub in candidates for i in sub]
        # update the vid_meta
        for key in self.vid_meta.keys():
            self.vid_meta[key] = [self.vid_meta[key][i] for i in candidates]

    # def _count(self):
    #     classes = 200
    #     count = [ 0 for _ in range(classes)]
    #     for hr, fps in zip(self.vid_meta["hr"], self.vid_meta["fps"]):
    #         count[int(hr*fps/25)]+=1
    #     x = [i for i in range(classes)]
    #     sns.set_style("whitegrid")
    #     g = sns.barplot(x = x[40:190], y=count[40:190], color="b")
    #     g.set_xticklabels(g.get_xticklabels(),rotation=90, size=6)
    #     plt.show()
        # for i in range(classes):
        #     if count[i]!=0:
        #         print(i)
        #         break
        # for i in reversed(range(classes)):
        #     if count[i]!=0:
        #         print(i)
        #         break

    def __len__(self):
        return len(self.vid_meta["data_path"])
    
    def __getitem__(self,index):
        data = np.load(self.vid_meta["data_path"][index])
        fps = self.vid_meta["fps"][index]

        # scale the heart rate by fps
        scale_rate = fps/25
        hr = self.vid_meta["hr"][index]

        # [T, RGB] -> [RGB, T]
        rgb_data = data["RGB_mean"].astype(np.float32).transpose()

        # if training, use randomly sample
        if self.mode == "train":
            try:
                _, sample_len = rgb_data.shape
                start = np.random.randint(0, sample_len-self.clip_len)
                rgb_data = rgb_data[:,start:start+self.clip_len]
            except:
                print(sample_len, self.vid_meta["vid_path"][index])

        return (
            np.expand_dims(rgb_data, axis=0), 
            int(hr*scale_rate-30), 
            torch.tensor(hr*scale_rate).float(), 
            scale_rate
        )

if __name__ == "__main__":
    
    dataset = RPPG_Dataset(label_path="/data/Heart-rate/train/cropped/valid-labels.txt", mode='val', cur=0)
    dataloader = DataLoader(dataset,1,shuffle=True,num_workers=8)

    for inputs, labels, scale_rate in dataloader:
        # print(inputs,labels,scale_rate)
        print(inputs.size())
        # pass
