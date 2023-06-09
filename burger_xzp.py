'''
python burger_xzp.py --gpu 5 --model ONO2  --n-hidden 128 --n-heads 2 --n-layers 8 --lr 0.001 --use_tb 1 --attn_type nystrom --max_grad_norm 0.1 --orth 1 --psi_dim 32 --batch-size 16
Time :7.2
rel_err:0.01092
'''
import argparse

parser = argparse.ArgumentParser(description="Demo of argparse for integer variables")



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
parser.add_argument('--downsample',type=int,default=32)
parser.add_argument("--use_tb", type=int, default=0, help="Use TensorBoard: 1 for True, 0 for False")
parser.add_argument("--gpu", type=str, default='1', help="GPU index to use")
parser.add_argument("--orth", type=int, default=0)
parser.add_argument("--psi_dim", type=int, default=64)
parser.add_argument('--attn_type',type=str, default=None)
parser.add_argument('--max_grad_norm',type=float, default=None)
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import scipy.io as scio
import gc
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ONOmodel import ONO
#from ONOori import ONO
from ONOmodel2 import ONO2
from timeit import default_timer
from tqdm import *
from data_utils import LossFunc as testLoss, UnitGaussianNormalizer, GaussianNormalizer, IdentityNormalizer, count_params
from torch.utils.tensorboard import SummaryWriter


# data_path = './burgers_data_R10.mat'
data_path = './data/burgers_data_R10.mat'

# a(2048, 8192) , u(2048, 8192)

class BurgerDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 n_grid = 8192,
                 train_data = True,
                 train_len = 1000,
                 test_len = 100,
                 downsample = 1,

                 ):
        self.data_path = data_path
        self.n_grid = n_grid
        self.train_data = train_data
        self.train_len = train_len
        self.test_len = test_len
        self.downsample = downsample
        
        if self.data_path is not None:
            self._initialize()
    
    def _initialize(self):
        
        data = scio.loadmat(self.data_path)
        
        self.a = data['a']
        self.u = data['u']
        del data
        gc.collect()

        if self.train_data :
            self.a = self.a[:self.train_len]
            self.u = self.u[:self.train_len]
            
        else :
            self.a = self.a[-self.test_len:]
            self.u = self.u[-self.test_len:]     
        
        self.pos = torch.from_numpy(np.linspace(0, 1, self.n_grid))[::self.downsample]
        self.a = torch.from_numpy(self.a)[:,::self.downsample]
        self.u = torch.from_numpy(self.u)[:,::self.downsample]
        
    def __len__(self):
        return self.a.shape[0]
    
    def __getitem__(self, index):
        pos = self.pos
        fx = self.a[index]
        u = self.u[index]

        return dict(pos=pos.float(),
                    fx = fx.float(),
                    u = u.float())
          
ntrain = 1000
ntest = 100

def main():
    train_dataset = BurgerDataset(data_path = data_path, train_len=ntrain,train_data = True, downsample=args.downsample)
    test_dataset = BurgerDataset(data_path = data_path, test_len=ntest,train_data = False,downsample=args.downsample)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print('dataloading is over')
    
    if args.model == 'ONO2':
        model = ONO2(n_hidden=args.n_hidden, n_layers=args.n_layers, space_dim=1, n_head = args.n_heads, attn_type=args.attn_type, orth=args.orth, psi_dim=args.psi_dim).cuda()
    else:
        raise NotImplementedError
        
    #model = ONO(n_hidden=args.n_hidden, n_layers=args.n_layers, space_dim=1, n_head = args.n_heads,res = 256).cuda()    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_writer = args.use_tb
    if use_writer:
        writer = SummaryWriter(log_dir='./logs/' + args.model + time.strftime('_%m%d_%H_%M_%S'))
    else:
        writer = None
    print(model)

    # OneCycleLR在中后段更稳定
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = testLoss(size_average=False)

    ## normalization
    # normalizer = UnitGaussianNormalizer(train_dataset.u)
    #normalizer = GaussianNormalizer(train_dataset.u)
    # normalizer = IdentityNormalizer(train_dataset.u)
    #train_dataset.u, test_dataset.u = normalizer.encode(train_dataset.u), normalizer.encode(test_dataset.u)

    #normalizer.cuda()
    
    for ep in range(args.epochs):

        model.train()
        t1 = default_timer()
        train_mse = 0
  
        for ii, data in enumerate(train_loader):
            x, fx , y = data['pos'].cuda(), data['fx'].cuda() , data['u'].cuda() 
            x = x.unsqueeze(-1)
            fx = fx.unsqueeze(-1)
            optimizer.zero_grad()
            out = model(x , fx).squeeze()
            # out = model(fx).squeeze()

            #ori_pred,  ori_target = normalizer.decode(out), normalizer.decode(y)
            loss = myloss(out, y)
            #ori_loss = myloss(ori_pred, ori_target)
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) 
            optimizer.step()
            train_mse+=loss.item()
            scheduler.step()

            #if ii % args.epochs == (args.epochs-1):
                #print("loss:{} ori loss {}".format(loss.item(), ori_loss.item()))

        train_mse = train_mse/ntrain
        model.eval()
        rel_err = 0
        testloss = testLoss(size_average=False)

        with torch.no_grad():
            for data in test_loader:
                x, fx, y = data['pos'].cuda(), data['fx'].cuda(), data['u'].cuda()
                x = x.unsqueeze(-1)
                fx = fx.unsqueeze(-1)

                out = model(x, fx).squeeze()
                # out = model(fx).squeeze()

                #ori_pred, ori_target = normalizer.decode(out), normalizer.decode(y)

                # rel_err+=testloss(out , y).item()
                rel_err += testloss(out, y).item()
            rel_err = rel_err / ntest
            print("The loss in epoch {}: {:.5f} test rel err {} lr {}".format(ep, train_mse, rel_err, optimizer.param_groups[0]['lr']))


    t2 = default_timer()
    print("Time :{:.1f}".format(t2-t1))
    print("rel_err:{:.5f}".format(rel_err))
       


if __name__ == "__main__":
    main()
