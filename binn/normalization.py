#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:31:02 2024

@author: lixiao
"""

import torch
from torch import nn
from typing import Union, List

class BatchNorm(nn.Module):
  
  def __init__(self, channels, eps = 1e-5, momentum = 0.1,
               affine = True, track_runing_stats = True):
    super().__init__()
    
    self.channels = channels
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.track_runing_stats = track_runing_stats
    
    if self.affine:
      self.scale = nn.Parameter(torch.ones(channels))
      self.shift = nn.Parameter(torch.zeros(channels))
      
    if self.track_runing_stats:
      self.register_buffer('exp_mean', torch.zeros(channels))
      self.register_buffer('exp_var', torch.ones(channels))
      
  def forward(self, x):
    
    assert self.channels == x.shape[1]
    batch_size = x.shape[0]
    orin_shape = x.shape
    
    x = x.view(batch_size, self.channels, -1)
    
    if self.training or not self.track_runing_stats:
      mean = x.mean(dim=[0, 2])
      mean_2 = (x**2).mean(dim=[0, 2])
      var = mean_2 - mean**2
      
      if self.training and self.track_runing_stats:
        self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
        self.exp_var  = (1 - self.momentum) * self.exp_var + self.momentum * var
    else:
      mean = self.exp_mean
      var  = self.exp_var
    
    x_norm = (x - mean.view(1, -1, 1))/torch.sqrt(var + self.eps).view(1, -1, 1)
    
    if self.affine:
      x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)
    
    return x_norm.view(orin_shape)
    

def __test_bn():
  x = torch.rand(2,3,2,4)
  
  bn = nn.BatchNorm2d(3, affine=True)
  lbn = BatchNorm(3, affine=True)
  
  o = bn(x)
  lo = lbn(x)
  print(((o - lo) < 1e-6).all())


class LayerNorm(nn.Module):
  
  def __init__(self, normalized_shape: Union[int, List[int], torch.Size], *,
               eps=1e-5, elementwise_affine=True):
    super().__init__()
    if isinstance(normalized_shape, int):
      normalized_shape = torch.Size([normalized_shape])
    elif isinstance(normalized_shape, list):
      normalized_shape = torch.Size(normalized_shape)
      
    self.normalized_shape = normalized_shape
    self.eps = eps
    self.elementwise_affine = elementwise_affine
    
    if self.elementwise_affine:
      self.gain = nn.Parameter(torch.ones(normalized_shape))
      self.bias = nn.Parameter(torch.zeros(normalized_shape))
      
  def forward(self, x):
    
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):]
    
    dims = [-(i + 1) for i in range(len(self.normalized_shape))]
    mean = x.mean(dim=dims, keepdim=True)
    mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)
    var = mean_x2 - mean ** 2
    x_norm = (x - mean) / torch.sqrt(var + self.eps)
    
    if self.elementwise_affine:
      x_norm = self.gain * x_norm + self.bias

    return x_norm

def __test_layernorm():
  # NLP Example
  batch, sentence_length, embedding_dim = 20, 5, 10
  embedding = torch.randn(batch, sentence_length, embedding_dim)
  layer_norm = nn.LayerNorm(embedding_dim)
  ln = LayerNorm(embedding_dim)
  # Activate module
  nlp_out = layer_norm(embedding)
  lo = ln(embedding)
  print((torch.abs((nlp_out - lo)) < 1e-6).all())
  
  # Image Example
  N, C, H, W = 20, 5, 10, 10
  input = torch.randn(N, C, H, W)
  # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
  # as shown in the image below
  layer_norm = nn.LayerNorm([C, H, W])
  ln1 = LayerNorm([C, H, W])
  lo1 = ln1(input)
  output = layer_norm(input)
  
  print((torch.abs((output - lo1)) < 1e-6).all())


def BNTT(input_features, time_steps, eps=1e-5, momentum=0.1, affine=True):
  '''
  Original GitHub repo:
    https://github.com/Intelligent-Computing-Lab-Yale/
    BNTT-Batch-Normalization-Through-Time
  '''
  bntt = nn.ModuleList(
    [
      BatchNorm(
        input_features, eps=eps, momentum=momentum, affine=affine
      )
      for _ in range(time_steps)
    ]
  )
  
  return bntt
  
class tdBatchNorm(nn.Module):
  '''
  Ref: https://arxiv.org/abs/2011.05280
  '''
  def __init__(self, channels, eps = 1e-5, momentum = 0.1,
               affine = True, track_runing_stats = True):
    super().__init__()
    
    self.channels = channels
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.track_runing_stats = track_runing_stats
    
    if self.affine:
      self.scale = nn.Parameter(torch.ones(channels))
      self.shift = nn.Parameter(torch.zeros(channels))
      
    if self.track_runing_stats:
      self.register_buffer('exp_mean', torch.zeros(channels))
      self.register_buffer('exp_var', torch.ones(channels))  

  def forward(self, x):
    # [T,B,C,H,W]
    assert self.channels == x.shape[2]
    orin_shape = x.shape
    
    if self.training or not self.track_runing_stats:
      mean = x.mean(dim=[0, 1, 3, 4])
      mean_2 = (x**2).mean(dim=[0, 1, 3, 4])
      var = mean_2 - mean**2
      
      if self.training and self.track_runing_stats:
        self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
        self.exp_var  = (1 - self.momentum) * self.exp_var + self.momentum * var
    else:
      mean = self.exp_mean
      var  = self.exp_var
    
    x_norm = (x - mean.view(1, 1, -1, 1, 1))/torch.sqrt(var + self.eps).view(1, 1, -1, 1, 1)
    
    if self.affine:
      x_norm = self.scale.view(1, 1, -1, 1, 1) * x_norm + self.shift.view(1, 1, -1, 1, 1)
    
    return x_norm.view(orin_shape)




















































