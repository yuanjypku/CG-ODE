import sys
import torch
import torch.nn as nn
from einops import rearrange, reduce
from base_models import DiffeqSolver, Encoder, ODEFunc, SeqODE
if __name__ =='__main__':
    sys.path.append("/nvme/yuanjingyang/CG-ODE")
from lib.utils import print_parameters
from lib.likelihood_eval import  masked_gaussian_log_density, compute_loss
import lib.utils as utils




def create_ODE_model(args, input_dim, z0_prior,obsrv_std,device):
    # RNN encoder
    # TODO:用args来调整hls和nl和T2
    resem_dim = 64*2 # mu & std
    encoder =  Encoder(input_size=input_dim,
                output_size=resem_dim,
                hidden_layer_size=200,
                num_layers=5,
                dropout=0.2,).to(device)
    # ODE related
    fc1 = nn.Linear(64,200)
    fc2 = nn.Linear(200,64)
    ode_func = ODEFunc(fc1,fc2,dropout=args.dropout,device=device,args=args).to(device)
    # ODE solver
    diffeq_solver = DiffeqSolver(ode_func, args.solver, args=args, odeint_rtol=args.rtol, odeint_atol=args.atol, device=device)

    # Decoder
    decoder = nn.Sequential(nn.Linear(64,1), nn.Dropout(args.dropout)).to(device)

    ode_hidden_dim = 1
    model = SeqODE(ode_hidden_dim, encoder, decoder, diffeq_solver,z0_prior,device, obsrv_std=obsrv_std)
    return model


if __name__ =='__main__':
    pass