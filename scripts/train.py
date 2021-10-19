"""Script to train and save the model"""

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *
from model import BatchCodeNet
from dataloader import get_bad_dataset, get_good_dataset, split_indices

device = get_default_device()

# data preparation for training
total_good_dataset = get_good_dataset('good')
total_bad_dataset = get_bad_dataset('bad')
final_dataset = data.ConcatDataset([total_good_dataset, total_bad_dataset])
print(len(final_dataset))

pct = 0.2
seed = 42       
train_indices, val_indices = split_indices(len(final_dataset), pct, seed)

batch_size = 32
train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(final_dataset,batch_size,sampler=train_sampler)
val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(final_dataset,batch_size,sampler=val_sampler)

# Getting model instance
model = BatchCodeNet()

# Copying data to device
train_dl = DeviceDataLoader(train_loader, device)
val_dl = DeviceDataLoader(val_loader, device)
to_device(model, device)

model = to_device(model, device)

# Training and saving
history = []
num_epochs = 1
opt_func = torch.optim.Adam
lr = 0.00001
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func, history)

torch.save(model, 'model_with_more_ngs.pth')