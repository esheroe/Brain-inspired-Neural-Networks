#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:39:15 2024

@author: lixiao
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class ActFun(torch.autograd.Function):
  '''
   Ref: STBP - https://arxiv.org/abs/1706.02609
  '''
  thresh = 1 # neuronal threshold
  lens = 0.5 # hyper-parameters of approximate function
  
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return input.gt(ActFun.thresh).float()

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.clone()
    temp = abs(input - ActFun.thresh) < ActFun.lens
    return grad_input * temp.float()
alif_fun = ActFun.apply

class LIFFun(torch.autograd.Function):

  @staticmethod
  def forward(ctx, input, thresh = 1, lens = 0.5):
    ctx.save_for_backward(input)
    ctx.thresh = thresh
    ctx.lens = lens
    return input.gt(thresh).float()

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output.clone()
    temp = abs(input - ctx.thresh) < ctx.lens
    return grad_input * temp.float(), None, None

def act_fun(x, thresh = 0.5, lens = 0.5):
  return LIFFun.apply(x, thresh, lens)

def LIF_Neuron(x, mem, spike, decay=0.2):
  '''
   Ref: STBP - https://arxiv.org/abs/1706.02609
  '''
  mem = mem * decay * (1. - spike) + x
  spike = act_fun(mem)
  return spike, mem

def ALIF_Neuron(x, mem, spike, decay=1., d=0.7):
  
  mem = (1.-spike)*mem*decay
  spike = alif_fun(mem)+d*mem
  
  return spike, mem

def ALIF_Neuron2(x, mem, spike, decay=0.2, d=0.3):

  mem = mem * decay * (1. - spike) + x
  spike = act_fun(mem)+d*mem
  return spike, mem

def output_Neuron(inputs, mem, tau_m=0.4, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    d_mem = -mem  +  inputs
    mem = mem+d_mem*tau_m
    return mem














































































