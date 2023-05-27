import os

import numpy as np
import scipy.io as scio
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ONOmodel import ONO
from timeit import default_timer
from tqdm import *
from testloss import TestLoss
train_path = './../data/piececonst_r421_N1024_smooth1.mat'
test_path = './../data/piececonst_r421_N1024_smooth2.mat'


batch_size = 10
learning_rate = 0.001
epochs = 500
step_size = 50
gamma = 0.5
ntrain = 1000
ntest = 100

def count_parameters(model):
  total_params = 0
  for name, parameter in model.named_parameters():
      if not parameter.requires_grad: continue
      params = parameter.numel()
      total_params+=params
  print(f"Total Trainable Params: {total_params}")
  return total_params

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last # if the time dimension is the last dim


    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                    std = self.std[...,sample_idx] + self.eps # T*batch*n
                    mean = self.mean[...,sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

#sol(n,421,421) coeff(n,421,421)
class DarcyDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 n_grid = 421,
                 data_len=1000,
                 ):
        self.data_path = data_path
        self.n_grid = n_grid
        self.data_len = data_len
        
        if self.data_path is not None:
            self._initialize()
    
    def _initialize(self):
        
        data = scio.loadmat(self.data_path)
        
        self.x = data['coeff'][:self.data_len]
        self.x = self.x.reshape(self.x.shape[0],-1,1)
        self.u = data['sol'][:self.data_len]
        self.u = self.u.reshape(self.u.shape[0],-1,1)

        del data
        gc.collect()
        
        x = np.linspace(0, 1, self.n_grid)
        y = np.linspace(0, 1, self.n_grid)
        x, y = np.meshgrid(x, y)
        self.pos = np.c_[x.ravel(), y.ravel()]
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        pos = torch.from_numpy(self.pos)
        fx = torch.from_numpy(self.x[index])
        u = torch.from_numpy(self.u[index])       
        
        return dict(pos=pos.float(),
                    fx = fx.float(),
                    u = u.float())

r = 5
h = int(((421 - 1)/r) + 1)
s = h
 
def main():
    train_data = scio.loadmat(train_path)
    
    x_train = train_data['coeff'][:ntrain,::r,::r][:,:s,:s]
    x_train = x_train.reshape(ntrain, -1)
    x_train = torch.from_numpy(x_train).float().cuda()
    y_train = train_data['sol'][:ntrain,::r,::r][:,:s,:s]
    y_train = y_train.reshape(ntrain, -1)
    y_train = torch.from_numpy(y_train).cuda()
    
    test_data = scio.loadmat(test_path)
    
    x_test = test_data['coeff'][:ntest,::r,::r][:,:s,:s]
    x_test = x_test.reshape(ntest, -1)
    x_test = torch.from_numpy(x_test).float().cuda()
    y_test= test_data['sol'][:ntest,::r,::r][:,:s,:s]
    y_test = y_test.reshape(ntest, -1)
    y_test = torch.from_numpy(y_test).cuda()

    x_normalizer = UnitGaussianNormalizer(x_train)
    y_normalizer = UnitGaussianNormalizer(y_train)
    x_normalizer.cuda()
    y_normalizer.cuda()
    
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_train = y_normalizer.encode(y_train)
    
    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).cuda().unsqueeze(0)
    pos_train = pos.repeat(ntrain,1,1)
    pos_test = pos.repeat(ntest,1,1)
    
    print("Dataloading is over.")
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test), batch_size=batch_size, shuffle=False)
    
    model = ONO(n_hidden = 128, n_layers=5, space_dim=2, ortho=True, res = s).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    #OneCycleLR在中后段更稳定
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = learning_rate, epochs=epochs, steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    count_parameters(model)  
    
    for ep in tqdm(range(epochs)):

        model.train()
        train_loss = 0
 
        for x, fx , y in train_loader:

            optimizer.zero_grad()
            out = model(x , fx.unsqueeze(-1)).squeeze(-1)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss = myloss(out, y)
            loss.backward()
            
            # print("loss:{}".format(loss.item()/batch_size))
            optimizer.step()
            train_loss+=loss.item()
            scheduler.step()            
        
        train_loss = train_loss/ntrain
        print("The loss in epoch{}:{:.5f}".format(ep, train_loss))


        model.eval()
        testloss = TestLoss(size_average=False)
        rel_err = 0.0
        with torch.no_grad():
            for x, fx, y in test_loader:
                
                out = model(x, fx.unsqueeze(-1)).squeeze(-1)
                out = y_normalizer.decode(out)

                tl = testloss(out, y).item()

                rel_err+=tl

        rel_err /= ntest
        print("rel_err:{}".format(rel_err))
       

if __name__ == "__main__":
    main()