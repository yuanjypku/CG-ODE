import sys
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from einops import rearrange, reduce, repeat

if __name__ == '__main__':
    sys.path.append("/nvme/yuanjingyang/CG-ODE")
from lib.base_models import VAE_Baseline
from lib import utils


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

class ODEFunc(nn.Module):
    def __init__(self,fc1,fc2, dropout,device = torch.device("cpu"),args=None ):
        super(ODEFunc, self).__init__()
        self.device = device
        self.fc1 = fc1
        self.fc2 = fc2
        self.nfe = 0
        self.dropout = nn.Dropout(dropout)
        self.heavyBall = args.heavyBall

        # Momentum parameters
        if self.heavyBall:
            self.gamma = utils.Parameter([args.gamma_guess], frozen=False)
            self.gammaact = nn.Sigmoid() if args.gamma_act == 'sigmoid' else args.gamma_act
            self.corr = utils.Parameter([args.corr], frozen=args.corrf)
            self.sp = nn.Softplus()
            self.sign = args.sign # Sign of df
            self.elem_t = None
            if args.actv_h is None:  # Activation for dh, GHBNODE only
                self.actv_h = nn.Identity()  
            elif args.actv_h == 'sigmoid':
                self.actv_h = nn.Sigmoid() 
            else:
                raise NotImplementedError
    def forward(self, t_local, z, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        z:  [K*N,D]
        """
        self.nfe += 1
        if not self.heavyBall:
            grad = self.get_diff_z(z)
        else:
            # print('debug:z.shape=',z.shape,'-'*80)  
            h, m = torch.split(z, z.shape[-2]//2, dim=-2)
            dh = self.actv_h(- m)
            dm = self.get_diff_z(h) * self.sign - self.gammaact(self.gamma()) * m
            dm = dm + self.sp(self.corr()) * h
            grad = torch.cat((dh, dm), dim=-2)
            if self.elem_t is None:
                # TODO:检查elem_t的用法
                pass
            else:
                grad = self.elem_t * grad

        return grad
    def get_diff_z(self,z):
        assert (not torch.isnan(z).any())
        grad = self.fc1(z)
        grad = self.dropout(grad)
        grad = nn.ReLU()(grad)
        grad = self.fc2(grad)
        grad = self.dropout(grad)
        grad = nn.ReLU()(grad)
        
        return grad

        




class DiffeqSolver(nn.Module,):
    def __init__(self, ode_func, method,args,
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.device = device
        self.ode_func = ode_func
        self.args = args

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict):
        '''
        :param first_point:  [K*N,D]
        :param time_steps_to_predict: [t]
        :return:
        '''
        if self.args.heavyBall:
            reczeros = torch.zeros_like(first_point)
            first_point = torch.cat([first_point, reczeros], dim=-2) 
        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)

        if self.args.heavyBall:
            pred_y, _ = torch.split(pred_y, pred_y.shape[-2]//2, dim=-2) #丢弃m只保留z

        return pred_y

class SeqODE(VAE_Baseline):
    def __init__(self, ode_hidden_dim, encoder_z0,decoder_node, diffeq_solver,z0_prior, device, obsrv_std=None):
        super(SeqODE, self).__init__(
			z0_prior=z0_prior,
			device=device, obsrv_std=obsrv_std
		)
        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder_node = decoder_node
        self.ode_hidden_dim =ode_hidden_dim
    
    def compute_all_losses(self, batch_dict_encoder,batch_dict_decoder,num_atoms,edge_lamda, kl_coef = 1.,istest=False):
        '''
        :param batch_dict_encoder:
        :param batch_dict_decoder: dict: 1. time 2. data: (K*N, T2, D)
        :param num_atoms:
        :param kl_coef:
        :return:
		'''
        pred_node, info,temporal_weights= self.get_reconstruction(batch_dict_encoder,batch_dict_decoder,num_atoms = num_atoms)
        # pred_node [ K*N , time_length, d]
        if istest:
            mask_index = batch_dict_decoder["masks"]
            pred_node = pred_node[:,mask_index,:]

		# KL divergence only contains node-level (only z_node are sampled, z_edge are computed from z_node)
        fp_mu, fp_std, fp_enc = info["first_point"]  # [K*N,D]
        fp_std = fp_std.abs()

        fp_distr = Normal(fp_mu, fp_std)
        assert(torch.sum(fp_std < 0) == 0.)
        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)  #[K*N,D_ode_latent]

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")
        
        if torch.isinf(kldiv_z0).any():
            locations = torch.where(kldiv_z0==float("inf"),torch.Tensor([1]).to(fp_mu.device),torch.Tensor([0]).to(fp_mu.device))
            locations = locations.to("cpu").detach().numpy()
            mu_locations = fp_mu.to("cpu").detach().numpy()*locations
            std_locations = fp_std.to("cpu").detach().numpy()*locations
            _,mu_values = utils.convert_sparse(mu_locations)
            _,std_values = utils.convert_sparse(std_locations)
            print(mu_values)
            print(std_values)
        
        kldiv_z0 = torch.mean(kldiv_z0)  # Contains infinity.

        #likelihood
        rec_likelihood = self.get_gaussian_likelihood(
			batch_dict_decoder["data"], pred_node,temporal_weights,
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

        loss = - torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0,0)
        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)



        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).data.item()
        results["MAPE"] = torch.mean(mape_node).data.item()
        results["MSE"] = torch.mean(mse_node).data.item()
        results["MAE"] = torch.mean(mae_node).data.item() 
        results["kl_first_p"] =  kldiv_z0.detach().data.item()
        results["std_first_p"] = torch.mean(fp_std).detach().data.item()

        # if istest:
        # 	print("Predicted Inc Deaths are:")
        # 	print(self.print_out_pred(pred_node,pred_edge))
        # 	print(self.print_out_pred_sum(pred_node))

        return results


    
    def get_reconstruction(self, batch_en,batch_de,num_atoms):

        #Encoder:
        batch_en = rearrange(batch_en,'b n t1 d -> t1 (b n) d').float()
        resem  = self.encoder_z0(batch_en)  # [batch*node, dim]
        first_point_mu, first_point_std = self.split_mean_mu(resem)

        first_point_enc = utils.sample_standard_gaussian(first_point_mu, first_point_std) #[K*N,D]

        first_point_std = first_point_std.abs()

        time_steps_to_predict = batch_de["time_steps"]

        assert (torch.sum(first_point_std < 0) == 0.)
        assert (not torch.isnan(time_steps_to_predict).any())
        assert (not torch.isnan(first_point_enc).any())

        assert (not torch.isnan(first_point_std).any())
        assert (not torch.isnan(first_point_mu).any())



        # ODE:Shape of sol_y #[ K*N + K*N*N, time_length, d], concat of node and edge.
        # K_N is the index for node.
        sol_y= self.diffeq_solver(first_point_enc,time_steps_to_predict)

        assert(not torch.isnan(sol_y).any())


        # Decoder:
        pred_node = self.decoder_node(sol_y)
        pred_node = rearrange(pred_node, "t2 b_n d -> b_n t2 d")


        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach()
        }

        return pred_node, all_extra_info, None

    def split_mean_mu(self,h):
        last_dim = h.size()[-1] //2
        res = h[:,:last_dim], h[:,last_dim:]
        return res