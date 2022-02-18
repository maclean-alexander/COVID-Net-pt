import os, argparse, pathlib
import time
import copy
import pandas as pd
import numpy as np
 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import warnings
warnings.filterwarnings('ignore')

from darwin_net.architectures.DarwinNet_groups4 import DarwinNetV2
from darwinai.torch.builder import build_model  # , BlockSpec, BuildMetrics
from darwinai.builder import BlockSpec
from darwin.enums.enums import BuildMetrics
 
from dataset import get_dataloader


NUM_CLASSES = 3
BATCH_SIZE = 16

 
parser = argparse.ArgumentParser(description='COVID-Net-CXR pytorch training script')
parser.add_argument('--datadir', default='../montefiore_severity/CXR')
parser.add_argument('--trainfile', default='../../alex.maclean/montefiore_severity/nov_29_split_80_10_10/train_mtf_sev.txt')
parser.add_argument('--validfile', default='../../alex.maclean/montefiore_severity/nov_29_split_80_10_10/valid_mtf_sev.txt')
parser.add_argument('--testfile', default='../../alex.maclean/montefiore_severity/nov_29_split_80_10_10/test_mtf_sev.txt')
parser.add_argument('--name', default='covid_net_model')
parser.add_argument('--geo', action='store_true', default=False)
parser.add_argument('--opc', action='store_true', default=False)
parser.add_argument('--modelfile', default=None)
parser.add_argument('--resnet', default=False)
 
args = parser.parse_args()

# Check which metric to use
measure = 'geo' if args.geo else 'opc' if args.opc else 'invalid'
if measure == 'invalid': raise ValueError
 

bin_map = np.array([[0.0,3.0], [3.0,6.0], [6.0,8.0]])
is_classification = True
 
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
 
# output path
output_path = './output/'
run_id = args.name + f'-{measure}'
run_path = output_path + run_id
pathlib.Path(run_path).mkdir(parents=True, exist_ok=True)
print('Output: ' + run_path)
 

train_dataloader = get_dataloader(
        csv_file=args.trainfile,
        datadir=args.datadir,
        batch_size=BATCH_SIZE,
        transform='base',
        bin_map=bin_map,
        is_classification=is_classification)
 
valid_dataloader = get_dataloader(
        csv_file=args.validfile,
        datadir=args.datadir,
        batch_size=1,
        transform=None,
        bin_map=bin_map,
        is_classification=is_classification) # no augmentations
 
dataloaders = {'train':train_dataloader, 'val':valid_dataloader}
dataset_sizes = {'train':len(train_dataloader)*BATCH_SIZE, 'val':len(valid_dataloader)}
 
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    '''taken from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html '''
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
 
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
 
        epoch_start = time.time()
 
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
 
            running_loss = 0.0
            running_corrects = 0
 
            for i, sample_batch in enumerate(dataloaders[phase]):
                inputs = sample_batch['image'].to(device, dtype=torch.float)
                # CXR sev to get OPC or GEO
                labels = sample_batch['label'][measure].to(device, dtype=torch.long)
                
                optimizer.zero_grad()
 
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
 
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
 
                if i % 100 == 0 and phase == 'train':
                    phase_batch = BATCH_SIZE if phase == 'train' else 1
                    print('Batch {} of {}'.format(i, dataset_sizes[phase]/phase_batch))
            if phase == 'train':
                scheduler.step()
 
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
 
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print('Epoch took {} seconds'.format(time.time() - epoch_start))
 
        print()
 
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
 
if args.resnet:
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    
    print(num_ftrs)

    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    # Load model weights
    if args.modelfile:
        print("Using model: " + str(args.modelfile))
        model_ft.load_state_dict(torch.load(os.path.join(args.modelfile, 'model')))
else:
    gsbuild_config = {
        "blockspecs": [
            BlockSpec(channels=40, depth=3),
            BlockSpec(channels=84, depth=4),
            BlockSpec(channels=176, depth=7),
            BlockSpec(channels=372, depth=3),
            ]
        }
    INPUT_SHAPE = [480, 480, 3]

    model_ft = DarwinNetV2(gsbuild_config['blockspecs'], INPUT_SHAPE, NUM_CLASSES)

model_ft = model_ft.to(device)
 
# initial basic training parameters
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
 
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
 
torch.save(model_ft.state_dict(), os.path.join(run_path, 'model'))
 

