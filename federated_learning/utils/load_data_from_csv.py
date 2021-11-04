import numpy as np
import pandas as pd
import torch as th
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import Resize

class PoisonDataset(Dataset):
    def __init__(self, train_csv_path, transform = None, isPoisoned = False, source_class=None, target_class=None):
        self.data = pd.read_csv(train_csv_path)
        self.transform = transform

        self.images = list()
        self.labels = list()
        
        for i in range(len(self.data)):
            self.images.append(self.transform(self.data.iloc[i, 1:].values.astype(np.uint8).reshape(1, 1024)))
            self.labels.append(self.data.iloc[i, 0])
        
        if isPoisoned:
            self.replace_X_with_Y(source_class, target_class)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.data)
    
    def replace_X_with_Y(self, source, target):
        for idx, label in enumerate(self.labels):
            if label == source:
                self.labels[idx] = target


def PoisonDataLoader(train_csv_path, batch_size, poisoned, source_class, target_class) -> DataLoader:
    transformer = transforms.Compose([
        # transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_set = PoisonDataset(train_csv_path, transformer, poisoned, source_class, target_class)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader

if __name__ == "__main__":
    CSV_PATH = '/home/fl/DataPoisoning_FL/poisoned_csv/poisoned_data.csv'
    train_loader = PoisonDataLoader(CSV_PATH, 100, True, 7, 2)
    imgs, labels = next(iter(train_loader))
    print(labels)
