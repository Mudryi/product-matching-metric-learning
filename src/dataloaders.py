from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import torch.utils.data
import pandas as pd


def cls_dataloaders(batch_size, num_workers, transform=None):
    train_data = datasets.ImageFolder('datasets/train', transform=transform['train'])
    valid_data = datasets.ImageFolder('datasets/eval', transform=transform['eval'])

    dataloaders = {'train': DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True),
                   'eval': DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)}
    return dataloaders


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, triplets_df, transform=None, loader=default_image_loader):
        self.triplets = triplets_df
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3 = self.triplets.iloc[index]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        img3 = self.loader(path3)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)


def triplet_dataloaders(batch_size, num_workers, transform=None):
    train_df = pd.read_csv('datasets/train_triplets.csv')
    eval_df = pd.read_csv('datasets/eval_triplets.csv')

    train_data = TripletImageLoader(train_df, transform=transform['train'])
    valid_data = TripletImageLoader(eval_df, transform=transform['eval'])

    print('Train images :', len(train_data))
    print('Valid images :', len(valid_data))

    dataloaders = {'train': DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True),
                   'eval': DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)}
    return dataloaders
