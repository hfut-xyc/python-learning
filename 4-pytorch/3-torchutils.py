import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

# class MyDataset(Dataset):
#     def __init__(self) -> None:
#         super().__init__()
#         self.data = np.zeros((20, 3))
#         self.label = np.ones(20)

#     def __getitem__(self, index):
#         print(index)
#         return self.data[index], self.label[index]

#     def __len__(self):
#         return len(self.data)

# def collate_fn(batch):
#     data_tuple, label_tuple = zip(*batch)    
#     datas = np.stack(data_tuple, 0)
#     labels = np.stack(label_tuple, 0)
#     return datas, labels

# dataset = MyDataset()
# dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn)

# num_batches = len(dataloader)
# datas, labels = next(iter(dataloader))

class MyDataSet(Dataset):
    def __init__(self):
        a = torch.tensor([1,2,3,4])
        b = torch.tensor([5,6,7])
        c = torch.tensor([7,8])
        d = torch.tensor([9])
        self.data = [a, b, c, d]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    batch.sort(key=lambda x: len(x), reverse=True)
    seq_len = [s.size(0) for s in batch]
    batch = pad_sequence(batch, batch_first=True)
    return batch, seq_len

embed = nn.Embedding(10, 3, padding_idx=0)
dataset = MyDataSet()
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

data, seq_len = next(iter(dataloader))
print(data)

data = embed(data)
print(data)

data = pack_padded_sequence(data, seq_len, batch_first=True)
print(data)
