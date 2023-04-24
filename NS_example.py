import numpy as np
import h5py
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ONOmodel import ONO
from timeit import default_timer
from tqdm import *

data_path = './ns_V1000_N100_T50_S128_64.mat'

# T(200,1) , a(100,64,64) , u(100,64,64,200)

class NSDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 n_grid = 64,
                 train_data = True,
                 train_len = 80,
                 test_len = 20,
                 time_step = 200
                 ):
        self.data_path = data_path
        self.n_grid = n_grid
        self.train_data = train_data
        self.train_len = train_len
        self.test_len = test_len
        self.time_step = time_step
        
        if self.data_path is not None:
            self._initialize()
    
    def _initialize(self):
        
        data = h5py.File(self.data_path)
        
        self.a = np.array(data['a']).swapaxes(0,2)
        self.u = np.array(data['u']).swapaxes(0,3).swapaxes(1,3)
        self.T = np.array(data['T'])
        del data
        gc.collect()
        
        if self.train_data :
            self.a = self.a[:self.train_len]
            self.u = self.u[:self.train_len]
            self.u = self.u.reshape(-1,self.u.shape[2],self.u.shape[3])
            
        else :
            self.a = self.a[-self.test_len:]
            self.u = self.u[-self.test_len:]
            self.u = self.u.reshape(-1,self.u.shape[2],self.u.shape[3])
            
        self.a = self.a.reshape(self.a.shape[0],-1)
        self.u = self.u.reshape(self.u.shape[0],-1)
        
        x = np.linspace(0, 1, self.n_grid)
        y = np.linspace(0, 1, self.n_grid)
        x, y = np.meshgrid(x, y)
        self.pos = np.c_[x.ravel(), y.ravel()]
        
    def __len__(self):
        return self.u.shape[0]
    
    def __getitem__(self, index):
        pos = torch.from_numpy(self.pos)
        fx = torch.from_numpy(self.a[index // self.time_step])
        u = torch.from_numpy(self.u[index])
        T = torch.from_numpy(self.T[index%self.time_step])
        
        return dict(pos=pos.float(),
                    fx = fx.float(),
                    u = u.float(),
                    T = T.float())
        
batch_size = 16
learning_rate = 0.001
epochs = 50
step_size = 100
gamma = 0.5      
        
def main():
    train_dataset = NSDataset(data_path = data_path, train_data = True)
    test_dataset = NSDataset(data_path = data_path, train_data = False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print('dataloading is over')
    
    model = ONO(space_dim=2, Time_Input= True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    myloss = torch.nn.MSELoss()

    for ep in tqdm(range(epochs)):

        model.train()
        t1 = default_timer()
        train_mse = 0
        n = 0
  
        for data in train_loader:
            x, fx , y , T = data['pos'].cuda(), data['fx'].cuda() , data['u'].cuda() , data['T'].cuda()
            optimizer.zero_grad()
            fx = fx.unsqueeze(-1)
            out = model(x , fx , T).squeeze(-1)
            loss = myloss(out, y)
            loss.backward()

            optimizer.step()
            train_mse+=loss
            n+=1
            print("loss:{}".format(loss.item()))
        
        train_mse = train_mse/n
        print("The loss in epoch{}:{:.5f}".format(ep, train_mse))
        scheduler.step()

    model.eval()
    rel_err = 0
    n = 0
    with torch.no_grad():
        for data in test_loader:
            x, fx , y , T = data['pos'].cuda(), data['fx'].cuda() , data['u'].cuda() , data['T'].cuda()

            out = model(x , fx , T)

            rel_err+=myloss(out , y)
            n+=1


    rel_err = rel_err/n
    t2 = default_timer()
    print("Time :{:.1f}".format(t2-t1))
    print("rel_err:{:.5f}".format(rel_err))
       
    torch.save(model.state_dict(), '/home/xzp/ONOtest/ONO/model/NS424.pkl')

if __name__ == "__main__":
    main()
        

