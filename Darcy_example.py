import numpy as np
import scipy.io as scio
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ONOmodel import ONO
from timeit import default_timer
import tqdm
train_path = './piececonst_r421_N1024_smooth1.mat'
test_path = './piececonst_r421_N1024_smooth2.mat'



batch_size = 20
learning_rate = 0.001
epochs = 500
step_size = 100
gamma = 0.5

class DarcyDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 n_grid = 4241
                 ):
        self.data_path = data_path
        self.n_grid = n_grid
        
        if self.data_path is not None:
            self._initialize()
    
    def _initialize(self):
        
        data = scio.loadmat(self.data_path)
        
        self.x = data['coeff'].reshape(data['coeff'].shape[0], -1, 1)
        self.u = data['sol'].reshape(data['sol'].shape[0] , -1 , 1)
        del data
        gc.collect()
        
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
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
    train_dataset = DarcyDataset(data_path = train_path)
    test_dataset = DarcyDataset(data_path = test_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = ONO(space_dim=2).cuda()
    print(model.count_params())

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    myloss = torch.nn.MSELoss()
    
    for ep in tqdm(range(epochs)):
        model.train()
        t1 = default_timer()
        train_mse = 0
        for data in train_loader:
            x, fx , y = data['pos'].cuda(), data['fx'].cuda() , data['u'].cuda()

            optimizer.zero_grad()
            out = model(x , fx)
            loss = myloss(out, y)
            loss.backward()

            optimizer.step()
            train_mse += loss
            
        print("The loss in epoch{}:{:.5f}".format(ep, train_mse))

    scheduler.step()

    model.eval()
    abs_err = 0.0
    rel_err = 0.0
    with torch.no_grad():
        for data in test_loader:
            x, fx , y = data['pos'].cuda(), data['fx'].cuda() , data['u'].cuda()

            out = model(x ,fx)

            rel_err += myloss(out , y)


    t2 = default_timer()
    print(ep, t2-t1, train_mse, rel_err)   


if __name__ == "__main__":
    main()