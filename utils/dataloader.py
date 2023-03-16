from torch.utils.data import DataLoader, Dataset, random_split
import torch
from torchvision import transforms
import numpy as np
from PIL import Image


class sampleDataset(Dataset):
    def __init__(self, df, use_transform=True) -> None:
        super().__init__()
        self.df = df
        if use_transform:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    
    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        path = 'data/MNIST/png/'
        '''获取第index份图像数据'''
        file = self.df.loc[index][0]
        class_id = self.df.loc[index][1]
        image = Image.open(path+file)
        if self.transform != None:
            image = self.transform(image) # image: Tensor(1, 28, 28)
        return image, int(class_id)
        # return torch.tensor(image, dtype=torch.float).unsqueeze(0), int(class_id) # (1, 28, 28)