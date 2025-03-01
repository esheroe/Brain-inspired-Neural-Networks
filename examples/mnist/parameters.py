#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:07:55 2024

@author: lixiao
"""

from dataclasses import dataclass, field 
import os
import torch

@dataclass
class Parameters():
    n_cpu: int =  4 if os.cpu_count() < 4 else os.cpu_count()
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed: int = 12345
    batch_size: int = 100
    test_batch_size: int = 200
    epochs: int = 800
    lr: float = 1e-3
    momentum: float = 0.9
    log_interval: int = 10
    
    # spiking nn settings
    thresh: float = 0.5 # neuronal threshold
    len: float = 0.5 # hyper-parameters of approximate function
    decay: float = 0.2 # decay constants
    step: int = 20

par = Parameters()