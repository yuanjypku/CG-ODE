import sys
import torch
import torch.nn as nn
from einops import rearrange, reduce
if __name__ =='__main__':
    sys.path.append("/nvme/yuanjingyang/CG-ODE")
from lib.utils import print_parameters
from lib.likelihood_eval import  masked_gaussian_log_density, compute_loss
import lib.utils as utils

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder,obsrv_std):
        super().__init__()
        self.obsrv_std =obsrv_std
        self.encoder = encoder
        self.decoder = decoder

    def compute_all_losses(self, batch_dict_encoder, batch_dict_decoder,num_atoms,edge_lamda, kl_coef=1.0,istest=False):
        '''
		:param batch_dict_encoder:
		:param batch_dict_decoder: dict: 1. time 2. data: (K*N, T2, D)
		:param num_atoms:
		:param kl_coef:
		:return:
		'''
        batch_dict_encoder = rearrange(batch_dict_encoder,'b n t1 d -> t1 (b n) d').float()
        resem  = self.encoder(batch_dict_encoder)  # [batch*node, dim]
        pred_node = self.decoder(resem, batch_dict_decoder['data'].shape[1]) # [T2, batch*node,dim]
        pred_node = rearrange(pred_node, "t2 b_n d -> b_n t2 d")

        # Compute likelihood of all the points
        rec_likelihood = self.get_gaussian_likelihood(
			batch_dict_decoder["data"], pred_node,None,
			mask=None)   #negative value

        mape_node = self.get_loss(
			batch_dict_decoder["data"], pred_node,truth_gt=batch_dict_decoder["data_gt"],
			mask=None,method = 'MAPE',istest = istest)  # [1]

        mse_node = self.get_loss(
			batch_dict_decoder["data"], pred_node,
			mask=None, method='MSE', istest=istest)  # [1]

        mae_node = self.get_loss(
			batch_dict_decoder["data"], pred_node,
			mask=None, method='MAE', istest=istest)  # [1]
        
        # loss
        loss = - torch.logsumexp(rec_likelihood,0)
        if torch.isnan(loss):
            loss = -torch.mean(rec_likelihood,0)
        
        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).data.item()
        results["MAPE"] = torch.mean(mape_node).data.item()
        results["MSE"] = torch.mean(mse_node).data.item()
        results["MAE"] = torch.mean(mae_node).data.item() 
        results["kl_first_p"] =  0
        results["std_first_p"] = 0

        return results

    def get_loss(self, truth, pred_y, truth_gt=None,mask = None,method='MSE',istest=False):
		# pred_y shape [n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]

		#Transfer from inc to cum

        truth = utils.inc_to_cum(truth)
        pred_y = utils.inc_to_cum(pred_y)
        num_times = truth.shape[1]
        time_index = [num_times-1] # last timestamp

        if istest:
            truth = truth[:,time_index,:]
            pred_y = pred_y[:,time_index,:]   #[N,1,D]
            if truth_gt != None:
                truth_gt = truth_gt[:,time_index,:]

		# Compute likelihood of the data under the predictions
        log_density_data = compute_loss(pred_y, truth, truth_gt,mask = mask,method=method)
		# shape: [1]
        return torch.mean(log_density_data)

    def get_gaussian_likelihood(self, truth, pred_y,temporal_weights=None, mask=None ):
		# pred_y shape [K*N, n_tp, n_dim]
		# truth shape  [K*N, n_tp, n_dim]

		# Compute likelihood of the data under the predictions

        log_density_data = masked_gaussian_log_density(pred_y, truth,
			obsrv_std = self.obsrv_std, mask = mask,temporal_weights= temporal_weights) #【num_traj = K*N] [250,3]
        log_density = torch.mean(log_density_data)

		# shape: [n_traj_samples]
        return log_density




class Encoder(nn.Module):
    def __init__(self, input_size,output_size,hidden_layer_size,num_layers,dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size,num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # TODO:记得加上dropout？
        # TODO:加上可选的random h0 c0，比较哪个学的好
        lstm_out, hidden_cell = self.lstm(input_seq,None) 
        ouput = self.linear(lstm_out)
        return ouput[-1,:,:] # 只保留序列尾端

class Decoder(nn.Module):
    def __init__(self, input_size,output_size,hidden_layer_size,num_layers,dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size,num_layers)
        self.fc = nn.Linear(hidden_layer_size, input_size)
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, z0, T2):
        z0 = z0.unsqueeze(0)
        output = []
        batch = z0.shape[-2]
        h,c = torch.randn(self.num_layers, batch, self.hidden_layer_size).to(z0.device), torch.randn(self.num_layers, batch, self.hidden_layer_size).to(z0.device)
        for i in range(T2):
            lstm_out, (h,c)= self.lstm(z0,(h,c))
            z0 = self.fc(lstm_out)
            output.append(z0)
        output = torch.cat(output, dim=0)
        output = self.linear(output)
        return output


def create_LSTM_model(args, input_dim,output_size,device):
    # TODO:用args来调整hls和nl和T2
    resem_dim = 64
    encoder =  Encoder(input_size=input_dim,
                output_size=resem_dim,
                hidden_layer_size=200,
                num_layers=5,
                dropout=0.2,).to(device)
    
    decoder = Decoder(input_size=resem_dim,
                output_size=output_size,
                hidden_layer_size=200,
                num_layers=5,
                dropout=0.2,).to(device)
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    model = Seq2seq(encoder,decoder,obsrv_std).to(device)
    print_parameters(model)
    return model



if __name__ == '__main__':
    model = create_LSTM_model(None, input_dim=7,device='cpu')
    t = torch.randn(100,10,7) # [T1, batch, dim]
    resem  = model.encoder(t)  # [batch, dim]
    model.decoder(resem) # [T2, batch,dim]
    print('end test')
