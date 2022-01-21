from torch.utils.data import Dataset
import cv2
import os
import glob
import numpy as np
import random
import torch
from collections import defaultdict
import sys
from torch.utils.data import DataLoader
# from turbojpeg import TurboJPEG


# jpeg = TurboJPEG()

class LRW1000_Dataset(Dataset):
    
    def __init__(self, index_file, target_dir):
        
        self.data = []
        self.index_file = index_file
        self.target_dir = target_dir
        lines = []

        with open(index_file, 'r') as f:
            lines.extend([line.strip().split(',') for line in f.readlines()])
        
                        
        self.data_root = 'D:/xyc/LRW1000_Public/images'
        self.padding = 40
        pinyins = sorted(np.unique([line[2] for line in lines]))

        self.data = [(line[0], int(float(line[3])*25)+1, int(float(line[4])*25)+1, pinyins.index(line[2])) for line in lines]
        max_len = max([data[2]-data[1] for data in self.data])
        data = list(filter(lambda data: data[2]-data[1] <= self.padding, self.data))                
        self.lengths = [data[2]-data[1] for data in self.data]
        self.pinyins = pinyins

        self.class_dict = defaultdict(list)

        for item in data:
                            
            self.class_dict[item[-1]].append(item)                

        self.data = []            
        for k, v in self.class_dict.items():
            n = len(v) 

            self.data.extend(v[:n])            


                                
    def __len__(self):
        return len(self.data)

    def load_video(self, item):
        #load video into a tensor
        (path, mfcc, op, ed, label) = item
        inputs, border = self.load_images(os.path.join(self.data_root, path), op, ed)        
                
        result = {}        
                
        result['video'] = inputs
        result['label'] = int(label)        
        result['duration'] = border.astype(np.bool)
        
        # savename = os.path.join(target_dir, f'{path}_{op}_{ed}.pkl')
        # torch.save(result, savename)
        
        return True

    def __getitem__(self, idx):

        r = self.load_video(self.data[idx])


        return r

    def load_images(self, path, op, ed):
        center = (op + ed) / 2
        length = (ed - op + 1)
        
        op = int(center - self.padding // 2)
        ed = int(op + self.padding)
        left_border = max(int(center - length / 2 - op), 0)
        right_border = min(int(center + length / 2 - op), self.padding)
        #print(length, center, op, ed, left_border, right_border)
        
        
        files =  [os.path.join(path, '{}.jpg'.format(i)) for i in range(op, ed)]
        files = filter(lambda path: os.path.exists(path), files)
        files = [cv2.imread(file) for file in files]
        files = [cv2.resize(file, (96, 96)) for file in files]        
        
        files = np.stack(files, 0)        
        t = files.shape[0]
        
        tensor = np.zeros((40, 96, 96, 3)).astype(files.dtype)
        border = np.zeros((40))
        tensor[:t,...] = files.copy()
        border[left_border:right_border] = 1.0
        
        # tensor = [jpeg.encode(tensor[_]) for _ in range(40)]
        
        return tensor, border


if(__name__ == '__main__'):
    
    # for subset in ['trn', 'val', 'tst']:
    #     target_dir = f'LRW1000_Public_pkl_jpeg/{subset}'
    #     index_file =  f'LRW1000_Public/info/{subset}_1000.txt'       

    #     if(not os.path.exists(target_dir)):
    #         os.makedirs(target_dir)    
        
    #     loader = DataLoader(LRW1000_Dataset(index_file, target_dir),
    #             batch_size = 96, 
    #             num_workers = 16,   
    #             shuffle = False,         
    #             drop_last = False)
        
    #     import time
    #     tic = time.time()
    #     for i, batch in enumerate(loader):
    #         toc = time.time()
    #         eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
    #         print(f'eta:{eta:.5f}')

    line = ['0785ed06ebd1832deead51b8caa700d8', '可以', 'ke yi', 2.30, 2.46]
    data = (line[0], int(float(line[3])*25)+1, int(float(line[4])*25)+1, 1)
    print(data)
    op = data[1]
    ed = data[2]
    center = (op + ed) / 2
    length = (ed - op + 1)
    
    op = int(center - 40 // 2)
    ed = int(op + 40)
    left_border = max(int(center - length / 2 - op), 0)
    right_border = min(int(center + length / 2 - op), 40)
    print(length, center, op, ed, left_border, right_border)