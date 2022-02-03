import os, argparse, pathlib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
from skimage import io, transform
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def get_dataloader(csv_file, datadir, batch_size=8, transform='base'):
    # TODO confirm image sizing
    rescale = transforms.Compose([Rescale((480, 480))])

    if transform == 'base':
        # perform basic training transforms
        transform = transforms.Compose(
                [rescale,
                 # TODO Add crop top transform
                 CentralCrop(),
                 ToTensor(),
                 RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15))])
    elif transform is None:
        # perform nothing but rescaling
        transform = transforms.Compose(
                [rescale,
                 ToTensor()])
    else:
        # Not sure if this is the right syntax??
        transform = transforms.Compose(
                [rescale,
                 transform])

    dataset = COVIDNetDataset(
            csv_file,
            datadir,
            transform)

    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)

    return dataloader


class COVIDNetDataset(Dataset):
    '''General dataset for COVIDNet'''

    def __init__(self, csv_file, datadir, transform=None):
        # TODO confirm that csv_file for CXR Sev data works with this
        self.df = pd.read_csv(csv_file, header=None, sep=' ')
        self.datadir = datadir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.datadir, self.df.iloc[idx, 1])
        image = io.imread(img_name)
        label = self.df.iloc[idx, 2]
        # TODO modify dict to fit GEO and OPC data
        sample = {'image':image, 'label':label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = transforms.ToTensor()(image)
        return {'image':image, 'label':label}

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image':img, 'label':label}

class CentralCrop(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        size = min(image.shape[0], image.shape[1])
        offset_h = int((image.shape[0] - size) / 2)
        offset_w = int((image.shape[1] - size) / 2)
        img = image[offset_h:offset_h + size, offset_w:offset_w + size]
        return {'image':img, 'label':label}

class RandomAffine(object):
    ''' Class to perform some augmentations from ImageDataGenerator in TF version '''

    def __init__(self, degrees, translate, scale):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        tform = transforms.RandomAffine(degrees=self.degrees,
                                        translate=self.translate,
                                        scale=self.scale)
        #image = transforms.ToTensor()(image)
        image = tform(image)
        return {'image':image, 'label':label}


