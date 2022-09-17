import torch.optim as optim
import math
from net import *
import util


class Trainer():
    def __init__(self,
                 model,
                 lrate,
                 wdecay,
                 clip,
                 step_size,
                 seq_out_len,
                 scaler,
                 device,
                 cl=True,
                 new_training_method=False):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lrate,
                                    weight_decay=wdecay)
        self.loss = util.masked_mae
        self.clip = clip
        self.step = step_size

        self.iter = 0
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl
        self.new_training_method = new_training_method

    def train(self,input, graph,real_val, idx=None, batches_seen=None):
        # input, ycl: [batch, D?, N?, T?]
        # real_value: [batch, N?, T?]
        self.iter += 1

        if self.iter % self.step == 0 and self.task_level < self.seq_out_len:
            self.task_level += 1
            if self.new_training_method:
                self.iter = 0

        self.model.train()
        self.optimizer.zero_grad()
        assert self.cl
        output = self.model(input,
                            graph,
                            idx=idx,
                            ycl=input,
                            batches_seen=self.iter,
                            task_level=self.task_level) # [K≡1, 1, N, 1]
        
        real = torch.unsqueeze(real_val, dim=1)
        predict = output.squeeze()
        
        ''
        loss = self.loss(predict,real, 0.0)
        mape = util.masked_mape(predict,real, 0.0).item()
        rmse = util.masked_rmse(predict,real, 0.0).item()  
        mae = util.masked_mae(predict,real, 0.0).item()
        
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        return loss.item(), mape, rmse, mae

    def eval(self, input,graph, real_val, ycl): 
        self.model.eval()
        with torch.no_grad():
            output = self.model(input,graph, ycl=input,task_level=self.task_level)
        
        predict = output.squeeze()
        real = torch.unsqueeze(real_val, dim=1)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        mae = util.masked_mae(predict,real, 0.0).item()
        return loss.item(), mape, rmse, mae