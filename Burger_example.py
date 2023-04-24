import numpy as np
import scipy.io as scio
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ONOmodel import ONO
from timeit import default_timer
from tqdm import *

data_path = './burgers_data_R10.mat'

# a(2048, 8192) , u(2048, 8192)

class BurgerDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 n_grid = 8192,
                 train_data = True,
                 train_len = 1024,
                 test_len = 128,
                 ):
        self.data_path = data_path
        self.n_grid = n_grid
        self.train_data = train_data
        self.train_len = train_len
        self.test_len = test_len
        
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
epochs = 50
step_size = 100
gamma = 0.5    

def main():
    train_dataset = BurgerDataset(data_path = data_path, train_data = True)
    test_dataset = BurgerDataset(data_path = data_path, train_data = False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print('dataloading is over')
    
    model = ONO(space_dim=1).cuda()
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
            x = x.unsqueeze(-1)
            fx = fx.unsqueeze(-1)
            optimizer.zero_grad()
            out = model(x , fx).squeeze()

            loss = myloss(out, y)
            loss.backward()
            print("loss:{}".format(loss.item()))
            optimizer.step()
            train_mse+=loss
            n+=1
            
        
        train_mse = train_mse/n
        print("The loss in epoch{}:{:.5f}".format(ep, train_mse))
        scheduler.step()

    model.eval()
    rel_err = 0
    n = 0
    with torch.no_grad():
        for data in test_loader:
            x, fx , y = data['pos'].cuda(), data['fx'].cuda() , data['u'].cuda() 

            out = model(x , fx )

            rel_err+=myloss(out , y)
            n+=1


    rel_err = rel_err/n
    t2 = default_timer()
    print("Time :{:.1f}".format(t2-t1))
    print("rel_err:{:.5f}".format(rel_err))
       
    torch.save(model.state_dict(), '/home/xzp/ONOtest/ONO/model/NS424.pkl')

if __name__ == "__main__":
    main()
