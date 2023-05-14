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

data_path = './burgers_data_R10.mat'

# a(2048, 8192) , u(2048, 8192)

class BurgerDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 n_grid = 8192,
                 train_data = True,
                 train_len = 1000,
                 test_len = 100,
                 sub = 1
                 ):
        self.data_path = data_path
        self.n_grid = n_grid//sub
        self.train_data = train_data
        self.train_len = train_len
        self.test_len = test_len
        self.sub = sub
        
        if self.data_path is not None:
            self._initialize()
    
    def _initialize(self):
        
        data = scio.loadmat(self.data_path)
        
        self.a = data['a'][:,::self.sub]
        self.u = data['u'][:,::self.sub]
        del data
        gc.collect()

        if self.train_data :
            self.a = self.a[:self.train_len]
            self.u = self.u[:self.train_len]
            
        else :
            self.a = self.a[-self.test_len:]
            self.u = self.u[-self.test_len:]     
        
        self.pos = np.linspace(0, 1, self.n_grid)
        
    def __len__(self):
        return self.a.shape[0]
    
    def __getitem__(self, index):
        pos = torch.from_numpy(self.pos)
        fx = torch.from_numpy(self.a[index])
        u = torch.from_numpy(self.u[index])
        
        return dict(pos=pos.float(),
                    fx = fx.float(),
                    u = u.float())
        
batch_size = 10
learning_rate = 0.001
epochs = 100
ntrain = 1000
ntest = 100

sub = 2**5 #subsampling rate
h = 2**13 // sub #total grid size divided by the subsampling rate
s = h

def main():
    train_dataset = BurgerDataset(data_path = data_path, train_len=ntrain,train_data = True, sub = sub)
    test_dataset = BurgerDataset(data_path = data_path, test_len=ntest,train_data = False, sub = sub)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print('dataloading is over')
    
    model = ONO(space_dim=1, ortho= True, n_hidden=128,res = s).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    myloss = TestLoss(size_average=False)
    
    print('s = {}'.format(s))
    
    for ep in tqdm(range(epochs)):

        model.train()
        t1 = default_timer()
        train_mse = 0
  
        for data in train_loader:
            x, fx , y = data['pos'].cuda(), data['fx'].cuda() , data['u'].cuda() 

            x = x.unsqueeze(-1)
            fx = fx.unsqueeze(-1)
            optimizer.zero_grad()
            out = model(x , fx)

            loss = myloss(out, y)
            loss.backward()
            print("loss:{}".format(loss.item()))
            optimizer.step()
            train_mse+=loss.item()
            
        
        train_mse = train_mse/ntrain
        print("The loss in epoch{}:{:.5f}".format(ep, train_mse))
        scheduler.step()

        model.eval()
        rel_err = 0
        testloss = TestLoss(size_average=False)
        with torch.no_grad():
            for data in test_loader:
                x, fx , y = data['pos'].cuda(), data['fx'].cuda() , data['u'].cuda() 
                x = x.unsqueeze(-1)
                fx = fx.unsqueeze(-1)
                out = model(x , fx )

                rel_err+=testloss(out , y).item()


        rel_err = rel_err/ntest
        t2 = default_timer()
        print("Time :{:.1f}".format(t2-t1))
        print("rel_err:{:.5f}".format(rel_err))
       


if __name__ == "__main__":
    main()
