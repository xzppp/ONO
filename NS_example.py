import os
import time
import argparse
parser = argparse.ArgumentParser('Training Transformer')

parser.add_argument('--lr',type=float, default=1e-3)
parser.add_argument('--epochs',type=int, default=500)
parser.add_argument('--weight_decay',type=float,default=1e-5)
parser.add_argument('--model',type=str, default='ONO2',choices=['FNO','ONO2','ONO','CGPT'])
parser.add_argument("--modes", type=int, default=12, help="Number of modes")
parser.add_argument("--width", type=int, default=32, help="Width")
parser.add_argument('--n-hidden',type=int, default=64, help='hidden dim of ONO')
parser.add_argument('--n-layers',type=int, default=3, help='layers of ONO')
parser.add_argument('--n-heads',type=int, default=4)
parser.add_argument('--batch-size',type=int, default=8)
parser.add_argument("--use_tb", type=int, default=0, help="Use TensorBoard: 1 for True, 0 for False")
parser.add_argument("--gpu", type=str, default='1', help="GPU index to use")
parser.add_argument("--orth", type=int, default=0)
parser.add_argument("--psi_dim", type=int, default=64)
parser.add_argument('--attn_type',type=str, default=None)
parser.add_argument('--max_grad_norm',type=float, default=None)
parser.add_argument('--downsample',type=int,default=1)
parser.add_argument('--momentum',type=float, default=0.9)
parser.add_argument('--mlp_ratio',type=int, default=1)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import scipy.io as scio
import numpy as np
import h5py
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from timeit import default_timer
from tqdm import *
from testloss import TestLoss

from ONOmodel import ONO
from ONOmodel2 import ONO2
from cgpt import CGPTNO

from torch.utils.tensorboard import SummaryWriter

data_path = './../data/NavierStokes_V1e-5_N1200_T20.mat'
ntrain = 1000
ntest = 200

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

class IdentityTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True) + 1e-8


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        return x

    def decode(self, x):
        return x


class UnitTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=(0,1), keepdim=True)
        self.std = X.std(dim=(0,1), keepdim=True) + 1e-8


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        return x* self.std + self.mean


    def transform(self, X, inverse=True,component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = X.shape
                return (X*(self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (X*(self.std[:,component] - 1e-8)+ self.mean[:,component]).view(orig_shape)
            else:
                return (X - self.mean[:,component])/self.std[:,component]



class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last  # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                std = self.std[..., sample_idx] + self.eps  # T*batch*n
                mean = self.mean[..., sample_idx]
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



def main():
    r = args.downsample
    h = int(((64 - 1) / r) + 1)
    T_in = 10
    T = 10
    step = 1
    
    data = scio.loadmat(data_path)

    train_a = data['a'][:ntrain, ::r, ::r][:, :h, :h]
    train_a = train_a.reshape(train_a.shape[0], -1)
    train_a = torch.from_numpy(train_a)
    train_u = data['u'][:ntrain, ::r, ::r, T:T+T_in][:, :h, :h, :]
    train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
    train_u = torch.from_numpy(train_u)
    
    a_normalizer = UnitTransformer(train_a)
    y_normalizer = UnitTransformer(train_u)
    
    test_a = data['a'][-ntest:, ::r, ::r][:, :h, :h]
    test_a = test_a.reshape(test_a.shape[0], -1)
    test_a = torch.from_numpy(test_a)
    test_u = data['u'][-ntest:, ::r, ::r, T:T+T_in][:, :h, :h, :]
    test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
    test_u = torch.from_numpy(test_u)
    
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)
    train_u = y_normalizer.encode(train_u)
    
    a_normalizer.cuda()
    y_normalizer.cuda()

    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    #t = data['t'][T: T+T_in]
    #t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
    #t_train = t.repeat(ntrain,1)
    #t_test = t.repeat(ntest,1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_u),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u),
                                              batch_size=args.batch_size, shuffle=False)
    
    print("Dataloading is over.")    

    
    if args.model == 'FNO':
        from fno_hzk import FNO2d
        from fourier_neural_operator.Adam import Adam
        model = FNO2d(modes1=args.modes, modes2=args.modes, width=args.width, res=s).cuda()
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.model in ['ONO', 'ONO2','CGPT']:
        if args.model == 'ONO2':
            model = ONO2(n_hidden=args.n_hidden, n_layers=args.n_layers, space_dim=2, Time_Input=True, n_head = args.n_heads, momentum=args.momentum, orth=args.orth, psi_dim=args.psi_dim, mlp_ratio=args.mlp_ratio, attn_type=args.attn_type).cuda()
        elif args.model == 'CGPT':
            model = CGPTNO(trunk_size=3, branch_sizes=None, output_size=1, n_layers=args.n_layers, n_hidden=args.n_hidden, n_inner=4, mlp_layers=2).cuda()
        else:
            raise NotImplementedError
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_writer = args.use_tb
    if use_writer:
        writer = SummaryWriter(log_dir='./logs/' + args.model + time.strftime('_%m%d_%H_%M_%S'))
    else:
        writer = None
        
    print(args)
    print(model)
    count_parameters(model)
    
    # OneCycleLR在中后段更稳定
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    for ep in range(args.epochs):

        model.train()
        train_l2_step = 0
        train_l2_full = 0
        
        for x, fx, yy in train_loader:
            loss = 0
            x, fx, yy  = x.cuda(), fx.cuda(), yy.cuda() # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
            bsz = x.shape[0]    
                    
            for t in range(0, T, step):
                y = yy[..., t:t+step]
                input_T = (torch.ones(bsz, 1)*t).cuda()
                #x = torch.cat((x,input_T),-1)
                im = model(x, fx.unsqueeze(-1), T = input_T)  #B , 4096 , 1
                #x = x[..., :-1]
                loss += myloss(im.reshape(bsz,-1), y.reshape(bsz,-1))
                
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                
            train_l2_step +=  loss.item()
            pred = y_normalizer.decode(pred)
            yy = y_normalizer.decode(yy)
            train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
            
            optimizer.zero_grad()
            loss.backward()
            # print("loss:{}".format(loss.item()/batch_size))
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

        test_l2_step = 0
        test_l2_full = 0

        model.eval()
        testloss = TestLoss(size_average=False)

        with torch.no_grad():
            for x, fx, yy in test_loader:
                loss = 0
                x, fx , yy  = x.cuda() ,fx.cuda(), yy.cuda() # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
                bsz = x.shape[0]    
                        
                for t in range(0, T, step):
                    y = yy[..., t:t+step]
                    input_T = (torch.ones(bsz, 1)*t).cuda()
                    #x = torch.cat((x,input_T),-1)
                    im = model(x, fx.unsqueeze(-1),T = input_T)
                    loss += testloss(im.reshape(bsz,-1), y.reshape(bsz,-1))
                    #x = x[..., :-1]                   
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    
                test_l2_step +=  loss.item()
                pred = y_normalizer.decode(pred)
                test_l2_full += testloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

        print("Epoch {} , train_step_loss:{:.5f} , train_full_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}".format(ep,  train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),test_l2_full / ntest))

if __name__ == "__main__":
    main()