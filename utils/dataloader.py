from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np
from PIL import Image


class sampleDataset(Dataset):
    def __init__(self, df) -> None:
        super().__init__()
        self.df = df

    
    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        path = 'data/MNIST/png/'
        '''获取第index份图像数据'''
        file = self.df.loc[index]['path']
        class_id = self.df.loc[index]['label']
        image = np.array(Image.open(path+file)) # image: (28, 28)
        return torch.tensor(image, dtype=torch.float).unsqueeze(0), int(class_id) # (1, 28, 28)