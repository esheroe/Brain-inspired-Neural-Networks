#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:27:28 2024

@author: lixiao
"""

import torch
import torch.nn as nn
from .neurons import ALIF_Neuron2, LIF_Neuron
from .normalization import BNTT

'''
  [T,B,C,H,W]
'''

class Linear(nn.Module):
  
  def __init__(self, *args, **kwargs):
    
    super().__init__()
    
    self.layer = nn.Linear(*args, **kwargs)
    
  def forward(self, x):
    # [T,B,C,H,W]
    x_ = []
    step = x.shape[0]
    for i in range(step):
      x_.append(self.layer(x[i]))
    return torch.stack(x_)


class Conv2d(nn.Module):
  
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.layer = nn.Conv2d(*args, **kwargs)
    
  def forward(self, x):
    # [T,B,C,H,W]
    x_ = []
    step = x.shape[0]
    for i in range(step):
      x_.append(self.layer(x[i]))
    return torch.stack(x_)

class AvgPool2d(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.layer = nn.AvgPool2d(*args, **kwargs)
  
  def forward(self, x):
    x_ = []
    step = x.shape[0]
    for i in range(step):
      x_.append(self.layer(x[i]))
    return torch.stack(x_)

class LIF_Spike(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    spk = torch.zeros(x.shape[1:] , device=x.device)
    mem = torch.zeros(x.shape[1:] , device=x.device)
    step = x.shape[0]
    
    for i in range(step):
      spk, mem = LIF_Neuron(x[i], mem, spk)
      x[i] = spk
    return x

class ALIF_Spike(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    spk = torch.zeros(x.shape[1:] , device=x.device)
    mem = torch.zeros(x.shape[1:] , device=x.device)
    step = x.shape[0]
    
    for i in range(step):
      spk, mem = ALIF_Neuron2(x[i], mem, spk)
      x[i] = spk
    return x
    
class BNTT_Layer(nn.Module):
  def __init__(self, input_features, step, **kwargs):
    super().__init__()
    self.step = step
    self.bntt = BNTT(input_features, step, **kwargs)

  def forward(self, x):
    for i in range(self.step):
      x[i] = self.bntt[i](x[i])
    return x

class Dropout(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.layer = nn.Dropout(*args, **kwargs)
  
  def forward(self, x):
    x_ = []
    step = x.shape[0]
    for i in range(step):
      x_.append(self.layer(x[i]))
    return torch.stack(x_)










































































