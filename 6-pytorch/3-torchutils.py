import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = np.zeros((20, 3))
        self.label = np.ones(20)

    def __getitem__(self, index):
        print(index)
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    data_tuple, label_tuple = zip(*batch)    
    datas = np.stack(data_tuple, 0)
    labels = np.stack(label_tuple, 0)
    return datas, labels

dataset = MyDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn)

num_batches = len(dataloader)
datas, labels = next(iter(dataloader))

# x = torch.tensor([1, 2, 3])
# y = torch.tensor([4, 5, 6])
# list = [x, y]
# print(list)
