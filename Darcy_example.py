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
train_path = './piececonst_r421_N1024_smooth1.mat'
test_path = './piececonst_r421_N1024_smooth2.mat'



batch_size = 5
learning_rate = 0.001
epochs = 5
step_size = 50
gamma = 0.5
ntrain = 1000
ntest = 1000


#sol(n,421,421) coeff(n,421,421)
class DarcyDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 n_grid = 421,
                 data_len=1000
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
        
def main():
    train_dataset = DarcyDataset(data_path = train_path, data_len=ntrain)
    test_dataset = DarcyDataset(data_path = test_path, data_len = ntest)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('dataloading is over')
    model = ONO(space_dim=2, ortho=True ).cuda()
    

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    myloss = torch.nn.MSELoss()
    
    for ep in tqdm(range(epochs)):

        model.train()
        t1 = default_timer()
        train_mse = 0
        n = 0
        for data in train_loader:
            x, fx , y = data['pos'].cuda(), data['fx'].cuda() , data['u'].cuda()

            optimizer.zero_grad()
            out = model(x , fx)
            loss = myloss(out, y)
            loss.backward()
            
            print("loss:{}".format(loss.item()))
            optimizer.step()
            train_mse+=loss.item()
            n+=1
        
        train_mse = train_mse/n
        print("The loss in epoch{}:{:.5f}".format(ep, train_mse))
        scheduler.step()

    model.eval()
    testloss = TestLoss(size_average=False)
    rel_err = 0.0
    with torch.no_grad():
        for data in test_loader:
            x, fx , y = data['pos'].cuda(), data['fx'].cuda() , data['u'].cuda()

            out = model(x ,fx)

            rel_err += testloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()


    rel_err /= ntest
    t2 = default_timer()
    
    print("rel_err:{}".format(rel_err))
       
    torch.save(model.state_dict(), '/home/xzp/ONOtest/ONO/model/darcy.pkl')

if __name__ == "__main__":
    main()