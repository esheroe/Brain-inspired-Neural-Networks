#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:44:31 2024

@author: lixiao
"""
import os
import sys
path = os.path.dirname(os.path.abspath(__file__)) + "/../../"
if path not in sys.path:
    sys.path.append(path)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from parameters import par
from torch.utils.data import default_collate
import binn as snn

class MNISTNet(nn.Module):  # Example net for MNIST
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.conv1 = snn.Conv2d(1, 32, 3, 1, 1)
    self.pool1 = snn.AvgPool2d(2)
    self.conv2 = snn.Conv2d(32, 32, 3, 1, 1)
    self.pool2 = snn.AvgPool2d(2)
    self.fc1 = snn.Linear(7 * 7 * 32, 300)
    self.fc2 = snn.Linear(300, 10)
    # self.spike = snn.LIF_Spike()
    self.spike = snn.ALIF_Spike()
    
    
  def forward(self, x):
    x = self.conv1(x) # 32, 28, 28
    x = self.spike(x) # 32, 28, 28
    x = self.pool1(x) # 32, 14, 14
    x = self.conv2(x) # 32, 14, 14
    x = self.spike(x) # 32, 14, 14
    x = self.pool2(x) # 32, 7, 7
    x = x.view(x.shape[0], x.shape[1], -1)
    x = self.fc1(x) # b, 7*7*32, 300
    x = self.fc2(x) # b, 300
    out = torch.mean(x, dim=0)
    return out

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  criterion = nn.CrossEntropyLoss()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)

    # necessary for general dataset: broadcast input
    output = model(data)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if batch_idx % par.batch_size == 0:
      print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
          epoch, batch_idx,len(train_loader.dataset)//par.batch_size, 
          loss.item()))
      
def test(model, device, test_loader, epoch):
  model.eval()
  test_loss = 0
  correct = 0
  criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)

      output = model(data)
      test_loss += criterion(output, target).item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def collate_fn(batch):
  inputs, targets = default_collate(batch)
  x = inputs 
  # [B C H W] -> [T B C H W]
  x = x.unsqueeze(0).expand(par.step, -1, -1, -1, -1).clone()
  return x, targets


device = par.device
torch.manual_seed(par.seed)
kwargs = {'num_workers': 8, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data/Datasets/Tiny', train=True, download=False,
                     transform=transforms.Compose([
                         transforms.ToTensor()
                     ])),
    batch_size=par.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data/Datasets/Tiny', train=False, 
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
    batch_size=par.test_batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)

model = MNISTNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=par.lr)

for epoch in range(30):
  train(model, device, train_loader, optimizer, epoch)
  test(model, device, test_loader, epoch)


















































