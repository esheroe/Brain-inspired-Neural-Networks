#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:42:13 2024

@author: lixiao
"""
import os
import sys
base_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_path, '../../') 
if path not in sys.path:
    sys.path.append(path)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import binn as snn


class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden = 128
    self.lstm1 = snn.ASLSTMCell(1, self.hidden)
    self.lstm2 = snn.ASLSTMCell(self.hidden, self.hidden, output=True)
    self.linear = nn.Linear(self.hidden , 1)

  def forward(self, input, future = 0):
    outputs = []
    h_t = torch.zeros(input.size(0), self.hidden, dtype=torch.float, device=input.device)
    c_t = torch.zeros(input.size(0), self.hidden, dtype=torch.float, device=input.device)
    s_t = torch.zeros(input.size(0), self.hidden, dtype=torch.float, device=input.device)
    h_t2 = torch.zeros(input.size(0), self.hidden, dtype=torch.float, device=input.device)
    c_t2 = torch.zeros(input.size(0), self.hidden, dtype=torch.float, device=input.device)
    s_t2 = torch.zeros(input.size(0), self.hidden, dtype=torch.float, device=input.device)
    
    max_h = []
    for input_t in input.split(1, dim=1):
      s_t, h_t, c_t = self.lstm1(input_t, (s_t, h_t, c_t))
      s_t2, h_t2, c_t2 = self.lstm2(s_t, (s_t, h_t2, c_t2))
      output = self.linear(h_t2)
      max_h.append(h_t)
      outputs += [output]
    for i in range(future):# if we should predict the future
      s_t, h_t, c_t = self.lstm1(output, (s_t, h_t, c_t))
      s_t2, h_t2, c_t2 = self.lstm2(s_t, (s_t, h_t2, c_t2))
      output = self.linear(h_t2)
      outputs += [output]
    outputs = torch.cat(outputs, dim=1)
    max_h = torch.cat(max_h)
    return outputs

class ANN_LSTM(nn.Module):
    def __init__(self):
        super(ANN_LSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 128)
        self.lstm2 = nn.LSTMCell(128, 128)
        self.linear = nn.Linear(128, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 128, dtype=torch.float, device=input.device)
        c_t = torch.zeros(input.size(0), 128, dtype=torch.float, device=input.device)
        h_t2 = torch.zeros(input.size(0), 128, dtype=torch.float, device=input.device)
        c_t2 = torch.zeros(input.size(0), 128, dtype=torch.float, device=input.device)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

def main():
    import argparse
    import torch.optim as optim
    
    folder = 'sin_pic'
    os.makedirs(folder, exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()
    np.random.seed(7)
    torch.manual_seed(7)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    
    seq = Net().to(device)
    # seq = ANN_LSTM().to(device)
    
    
    input = input.view(97, 999).float().to(device)
    target = target.view(97, 999).float().to(device)
    test_input = test_input.view(3, 999).float().to(device)
    test_target = test_target.view(3, 999).float().to(device)
    
   
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    
    
    #begin to train
    for i in range(opt.steps):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        torch.nn.utils.clip_grad_norm_(seq.parameters(), 1.0)
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 999
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().cpu().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig(folder+'/predict%d.pdf'%i)
        plt.close()
    
if __name__ == '__main__':
  main()
