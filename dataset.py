import os, argparse, pathlib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
 
import pandas as pd
from skimage import io, transform
import numpy as np
 
import warnings
warnings.filterwarnings('ignore')
 
def get_dataloader(csv_file, datadir, batch_size=8, transform='base', is_classification=False, bin_map=None):
 
    rescale = transforms.Compose([Rescale((480, 480))])
 
    if transform == 'base':
        # perform basic training transforms
        transform = transforms.Compose(
                [rescale,
                 # crop top transform
                 CropTop(percent=0.15),
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
            transform,
            bin_map=bin_map,
            is_classification=is_classification)
 
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)
            
 
    return dataloader
 
 
class COVIDNetDataset(Dataset):
    '''General dataset for COVIDNet'''
 
    def __init__(self, csv_file, datadir, transform=None, bin_map=None, is_classification=False):
        # TODO confirm that csv_file for CXR Sev data works with this
        self.df = pd.read_csv(csv_file, header=None, sep=' ')
        self.datadir = datadir
        self.transform = transform
        self.bin_map = bin_map
        self.is_classification = is_classification
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
 
        # For CXR, image file is at index 0
        img_name = os.path.join(self.datadir, self.df.iloc[idx, 0])
        image = io.imread(img_name)
       
        # For CXR, geo label is at index 2 and opc label is at index 3
        geo_label = self.df.iloc[idx, 2]
        opc_label = self.df.iloc[idx, 3]

        # If classification, assign labels to appropriate bins
        if self.is_classification:
            geo_label = _categorize_severity(geo_label, bin_map=self.bin_map)
            opc_label = _categorize_severity(opc_label, bin_map=self.bin_map)

    
        #Dict holds GEO and OPC data
        sample = {'image':image, 'label':{'geo': geo_label, 'opc': opc_label}}
 
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
 

class CropTop(object):
    ''' Class to perform croptop augmentation from TF version '''
 
    def __init__(self, percent=0.15):
        self.percent = percent

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        offset = int(image.shape[0] * self.percent)
        img = image[offset:]
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
 
 
 
# Assign class value to severity score based on bin sizes
def _categorize_severity(score, bin_map):
    score = float(score)
    if (score < 0.0) or (score > 8.0):
            raise ValueError('Scores must be between 0.0 and 8.0')

    for i in np.arange(bin_map.shape[0]):
        if bin_map[i,0] > bin_map[i,1]:
            raise ValueError('Bin boundaries must be in format:[left bound, right bound]')
        if i > 0:
            if bin_map[i,0] < bin_map[i-1,1]:
                ValueError('Bins must not overlap')
    category = None
    for i in np.arange(bin_map.shape[0]):
        if bin_map[i,0] <= score and score < bin_map[i,1]:
            category = i
            return category
    if score == bin_map[-1,1]:
            category = bin_map.shape[0] - 1
            return category
    
    print(score)
    raise ValueError('Scores must be between 0.0 and 8.0')