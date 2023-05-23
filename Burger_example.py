import argparse

parser = argparse.ArgumentParser(description="Demo of argparse for integer variables")



parser.add_argument('--lr',type=float, default=1e-3)
parser.add_argument('--epochs',type=int, default=1000)
parser.add_argument('--weight_decay',type=float,default=1e-4)
parser.add_argument('--downsample',type=int,default=4)
parser.add_argument("--modes", type=int, default=12, help="Number of modes")
parser.add_argument("--width", type=int, default=64, help="Width")
parser.add_argument('--batch-size',type=int, default=20)
parser.add_argument("--use_tb", type=int, default=1, help="Use TensorBoard: 1 for True, 0 for False")
parser.add_argument("--gpu", type=str, default='3', help="GPU index to use")

args = parser.parse_args()


import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import scipy.io as scio
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ONO.ONOmodel import ONO
from timeit import default_timer
from tqdm import *
from data_utils import LossFunc as testLoss, UnitGaussianNormalizer, GaussianNormalizer, IdentityNormalizer, count_params

from fourier_neural_operator.fourier_1d import FNO1d
from fourier_neural_operator.Adam import Adam as CAdam

# data_path = './burgers_data_R10.mat'
data_path = './../data/burgers_data_R10.mat'

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
        
batch_size = 10
learning_rate = 0.001
epochs = 50
step_size = 100
gamma = 0.5    
ntrain = 1000
ntest = 100
print_freq = 20

def main():
    train_dataset = BurgerDataset(data_path = data_path, train_data = True, downsample=args.downsample)
    test_dataset = BurgerDataset(data_path = data_path, train_data = False,downsample=args.downsample)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print('dataloading is over')
    
    model = ONO(space_dim=1,f_dim=1,n_layers=4,n_hidden=128, ortho= False).cuda()
    # model = FNO1d(modes=12, width=32).cuda()

    print('num params {}'.format(count_params(model)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = CAdam(model.parameters(),lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # myloss = torch.nn.MSELoss()
    myloss = testLoss(size_average=True)

    ## normalization
    # normalizer = UnitGaussianNormalizer(train_dataset.u)
    normalizer = GaussianNormalizer(train_dataset.u)
    # normalizer = IdentityNormalizer(train_dataset.u)
    train_dataset.u, test_dataset.u = normalizer.encode(train_dataset.u), normalizer.encode(test_dataset.u)

    normalizer.cuda()
    
    for ep in range(epochs):

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

            ori_pred,  ori_target = normalizer.decode(out), normalizer.decode(y)
            loss = myloss(out, y)
            ori_loss = myloss(ori_pred, ori_target)
            loss.backward()

            optimizer.step()
            train_mse+=loss.item()
            if ii % epochs == (epochs-1):
                print("loss:{} ori loss {}".format(loss.item(), ori_loss.item()))


        train_mse = train_mse/len(train_loader)

        scheduler.step()
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

                ori_pred, ori_target = normalizer.decode(out), normalizer.decode(y)

                # rel_err+=testloss(out , y).item()
                rel_err += testloss(ori_pred, ori_target).item()
            rel_err = rel_err / ntest
            print("The loss in epoch {}: {:.5f} test rel err {}".format(ep, train_mse, rel_err))


    t2 = default_timer()
    print("Time :{:.1f}".format(t2-t1))
    print("rel_err:{:.5f}".format(rel_err))
       
    torch.save(model.state_dict(), '/home/xzp/ONOtest/ONO/model/Burger424.pkl')

if __name__ == "__main__":
    main()
