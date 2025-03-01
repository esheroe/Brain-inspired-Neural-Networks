#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:15:42 2024

@author: lixiao
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from .neurons import (
    output_Neuron,
    ALIF_Neuron
  )

class ASLSTMCell(nn.Module):
  
  def __init__(self, input_size, hidden_size, *, 
               act_fun=None, layer_norm = False, output=False):
    super().__init__()
    
    self.output = output
    self.hidden_lin = nn.Linear(hidden_size, 4*hidden_size)
    self.input_lin  = nn.Linear(input_size, 4*hidden_size, bias=False)
    
    self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
    self.layer_norm_c = nn.Identity()
    if layer_norm:
      # layer norm
      pass
    
    if act_fun is None:
      self.act_fun = ALIF_Neuron
    else:
      self.act_fun = act_fun
    
  def forward(self, x, state):
    s, h, c = state
    ifgo = self.hidden_lin(h) + self.input_lin(x)
    ifgo = ifgo.chunk(4, dim=-1)
    
    ifgo = [self.layer_norm[i](ifgo[i]) for i in range(4)]
    
    i, f, g, o = ifgo
    c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))
    
    if self.output:
      h_next = output_Neuron(x, h_next)
    else:
      s, h_next = self.act_fun(s, h_next, s)
    
    return s, h_next, c_next

class ASLSTM(nn.Module):
  
  def __init__(self, input_size, hidden_size, n_layers = 1, batch_first=True,
               ):
    
    super().__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.batch_first = batch_first
    self.input_size = input_size
    
    self.cells = nn.ModuleList(
        [ASLSTMCell(input_size, hidden_size)] +
        [ASLSTMCell(hidden_size, hidden_size) for _ in range(n_layers - 1)]
      )
    self.cells[-1].output = True
    
  def forward(self, x, state=None):
    
    if self.batch_first:
      batch_size, n_steps = x.shape[:2]
    else:
      n_steps, batch_size = x.shape[:2]
    
    if state is None:
      s = x.new_zeros((self.n_layers, batch_size, self.hidden_size))
      h = x.new_zeros((self.n_layers, batch_size, self.hidden_size))
      c = x.new_zeros((self.n_layers, batch_size, self.hidden_size))
    else:
      (s, h, c) = state
    
    dim = 0
    s, h, c = list(torch.unbind(s, dim=dim)), list(torch.unbind(h, dim=dim)), list(torch.unbind(c, dim=dim))
    out = []
    
    for t in range(n_steps):
      inp = x[:,t,:] if self.batch_first else x[t]
      for layer in range(self.n_layers):
        s[layer], h[layer], c[layer] = self.cells[layer](inp, (s[layer], h[layer], c[layer]))
        inp = s[layer]
        
      out.append(h[-1])
    
    out = torch.stack(out, dim=dim)
    s = torch.stack(s, dim=dim)
    h = torch.stack(h, dim=dim)
    c = torch.stack(c, dim=dim)
    
    if self.batch_first:
      out = out.permute(1, 0, 2)
    return out, (s, h, c)


def __test_batch_first():
  print('='*20)
  print('batch_first=False')
  model = nn.LSTM(input_size=6, hidden_size=20, num_layers=2, batch_first=False)
  model = model.double()
  l = ASLSTM(input_size=6, hidden_size=20, n_layers=2, batch_first=False)
  l.double()
  x = torch.randn(100, 10, 6).double()
  print(x.shape)
  
  y, (hn, cn) = model(x)  
  print('y:', y.shape)
  print('hn:', hn.shape)
  print('cn:', cn.shape)
  
  print('\nslstm:')
  y, (sn, hn, cn) = l(x)  
  print('y:', y.shape)
  print('sn:', sn.shape)
  print('hn:', hn.shape)
  print('cn:', cn.shape)
  print('\n')
  
  print('='*20)
  print('batch_first=True')
  model = nn.LSTM(input_size=6, hidden_size=20, num_layers=2, batch_first=True)
  model = model.double()
  l = ASLSTM(input_size=6, hidden_size=20, n_layers=2, batch_first=True)
  l.double()
  x = torch.randn(100, 10, 6).double()
  print(x.shape)
  
  y, (hn, cn) = model(x)  
  print('y:', y.shape)
  print('hn:', hn.shape)
  print('cn:', cn.shape)
  
  print('\nslstm:')
  y, (sn, hn, cn) = l(x)  
  print('y:', y.shape)
  print('sn:', sn.shape)
  print('hn:', hn.shape)
  print('cn:', cn.shape)


if __name__ == '__main__':
  __test_batch_first()
























