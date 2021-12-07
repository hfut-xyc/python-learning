import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = np.zeros((20, 3))
        self.label = np.ones(20)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    data_tuple, label_tuple = zip(*batch)  # (array([0., 0., 0.]), array([0., 0., 0.]), array([0., 0., 0.]), array([0., 0., 0.]))
    data = np.stack(data_tuple, 0)
    label = np.stack(label_tuple, 0)
    
    return data, label

dataset = MyDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn)
datas, labels = next(iter(dataloader))
num_batches = len(dataloader)

for data, label in dataloader:
    print(data.shape)