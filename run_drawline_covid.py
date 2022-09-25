import os
import sys
import time
import random
from lib.load_data_covid import ParseData
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import lib.utils as utils
from torch.distributions.normal import Normal
from lib.create_coupled_ode_model import create_CoupledODE_model
from lib.utils import test_data_covid
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser('Coupled ODE')

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('--dataset', type=str, default='Dec', help="Dec")
parser.add_argument('--datapath', type=str, default='data/', help="default data path")
parser.add_argument('--pred_length', type=int, default=14, help="Number of days to predict ")
parser.add_argument('--condition_length', type=int, default=21, help="Number days to condition on")
parser.add_argument('--features', type=str,
                    default="Confirmed,Deaths,Recovered,Mortality_Rate,Testing_Rate,Population,Mobility",
                    help="selected features")
parser.add_argument('--split_interval', type=int, default=3,
                    help="number of days between two adjacent starting date of two series.")
parser.add_argument('--feature_out', type=str, default='Deaths',
                    help="Confirmed, Deaths, or Confirmed and deaths")

parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-3, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=8)
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--edge_lamda', type=float, default=0.5, help='edge weight')

parser.add_argument('--z0-encoder', type=str, default='GTrans', help="GTrans")
parser.add_argument('--rec-dims', type=int, default= 64, help="Dimensionality of the recognition model .")
parser.add_argument('--ode-dims', type=int, default= 20, help="Dimensionality of the ODE func for edge and node (must be the same)")
parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in recognition model ")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers  ODE func ")

# encoder choice
parser.add_argument('--encoder','-e',type=str, default='base', help="CVPR, Cheb, Cheb_cat,...")
parser.add_argument('--graf_layer', type=int, default=1, help="(CVPR mode)layer of graformers")
parser.add_argument('--cheb_K', type=int, default=2, help="(Cheb mode)order of Cheblayer")

# momentum
parser.add_argument('--heavyBall', action='store_true', help="Wheather to use Momentum")
parser.add_argument('--actv_h', type=str, default=None, help="Activation for dh, GHBNODE only")
parser.add_argument('--gamma_guess', type=float,default=-3.0,)
parser.add_argument('--gamma_act', type=str, default='sigmoid', help="gamma action")
parser.add_argument('--corr', type=int, default=-100, help="corr")
parser.add_argument('--corrf', type=bool, default=True, help="corrf")
parser.add_argument('--sign', type=int, default=1, help="sign")

parser.add_argument('--augment_dim', type=int, default=0, help='augmented dimension')
parser.add_argument('--solver', type=str, default="rk4", help='dopri5,rk4,euler')
parser.add_argument('--rtol', type=float,default=1e-2, help='tolerance of ode')
parser.add_argument('--atol', type=float,default=1e-2, help='tolerance of ode adjent')

parser.add_argument('--alias', type=str, default="run")
args = parser.parse_args()


############ CPU AND GPU related
if torch.cuda.is_available():
	print("Using GPU" + "-"*80)
	device = torch.device("cuda:0")
else:
	print("Using CPU" + "-" * 80)
	device = torch.device("cpu")

###########  feature related:
if args.feature_out == "Confirmed":
    args.output_dim = 1
    args.feature_out_index = [0]
elif args.feature_out == "Deaths":
    args.output_dim = 1
    args.feature_out_index = [1]
else:
    args.output_dim = 2
    args.feature_out_index = [0, 1]


#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #Saving Path
    file_name = os.path.basename(__file__)[:-3]  # run_models
    utils.makedirs(args.save)
    # experimentID = int(SystemRandom().random() * 100000) 换个带时间的log名
    experimentID = time.strftime("%m-%d_%H:%M", time.localtime(time.time() ))

    #Command Log
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)


    #Loading Data
    print("predicting data at: %s" % args.dataset)
    dataloader = ParseData(args =args)
    train_encoder, train_decoder, train_graph, train_batch, num_atoms = dataloader.load_train_data(is_train=True)
    val_encoder, val_decoder, val_graph, val_batch, _ = dataloader.load_train_data(is_train=False)
    args.num_atoms = num_atoms
    input_dim = dataloader.num_features

    # Model Setup
    # Create the model
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    model = create_CoupledODE_model(args, input_dim, z0_prior, obsrv_std, device)

    # Load checkpoint for saved model
    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        utils.get_ckpt_model(ckpt_path, model, device)
        print("loaded saved ckpt!")
        #exit()

    def drawline_covid(model, pred_length, condition_length, dataloader,device,args,kl_coef):

        encoder, decoder, graph, num_batch = dataloader.load_test_data(pred_length=pred_length,
                                                                    condition_length=condition_length)

        with torch.no_grad():
            for batch in tqdm(range(num_batch)):
                batch_dict_encoder = utils.get_next_batch_new(encoder, device)
                batch_dict_graph = utils.get_next_batch_new(graph, device)
                batch_dict_decoder = utils.get_next_batch_test(decoder, device)

                pred_node, _, _, _= model.get_reconstruction(batch_dict_encoder,batch_dict_decoder,num_atoms = args.num_atoms)
                truth_node = batch_dict_decoder["data"]
                
                for city in range(len(truth_node)):
                    plt.figure()
                    plt.plot(truth_node[city].cpu(), 'b')
                    plt.plot(pred_node[city].cpu(), 'r')
                    plt.legend(['ground_truth', 'prediction'])
                    plt.savefig('./figs/batch%d_city%d.png'%(batch, city))
                    plt.close()

    # DrawLine once: for loaded model
    if args.load is not None:
        test_res, MAPE_each, RMSE_each, MAE_each = drawline_covid(model, args.pred_length, args.condition_length, dataloader,
                                                   device=device, args=args, kl_coef=0)

        # message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | MAPE {:.6F} | RMSE {:.6F} | MAE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
        #     0,
        #     test_res["loss"], test_res["MAPE"], test_res["RMSE"], test_res["MAE"], test_res["likelihood"],
        #     test_res["kl_first_p"], test_res["std_first_p"])
    else:
        exit("You Must Load a Model!")


