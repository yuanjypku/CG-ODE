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
from lstm.create_lstm_model import create_LSTM_model
# from lib.create_coupled_ode_model import create_CoupledODE_model
from lib.utils import test_data_covid

parser = argparse.ArgumentParser('Baselines')

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

parser.add_argument('--model', type=str, default='LSTM', help='LSTM, ...')

parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-3, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=8)
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
# parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')

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

    # do not save 
    # Command Log
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

    #Model Setup
    #create the model
    if args.model == 'LSTM':
        model = create_LSTM_model(args,)
    #do not load checkpoint

    #Training Setup
    experimentID = time.strftime("%m-%d_%H:%M", time.localtime(time.time()+8*60**2))
    log_path = "baseline_logs/" + args.alias +"_" + args.dataset +  "_Con_"  + str(args.condition_length) +  "_Pre_" + str(args.pred_length) + "_" + str(experimentID) + ".log"
    if not os.path.exists("baseline_logs/"):
        utils.makedirs("baseline_logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    logger.info(args.alias)

    # Optimizer
    if args.optimizer == "AdamW":
        optimizer =optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)

    wait_until_kl_inc = 10
    best_test_MAPE = np.inf
    best_test_RMSE = np.inf
    best_val_MAPE = np.inf
    best_val_RMSE = np.inf
    n_iters_to_viz = 1

    def train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef):\

        optimizer.zero_grad()
        # TODO:每个模型需定义
        train_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,args.num_atoms,edge_lamda = args.edge_lamda, kl_coef=kl_coef,istest=False)
        # 记录forward中的nfe
        # forward_nfe = model.diffeq_solver.ode_func.nfe
        # model.diffeq_solver.ode_func.nfe = 0
        loss = train_res["loss"]
        loss.backward()
        # backward_nfe = model.diffeq_solver.ode_func.nfe
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) #需要吗

        optimizer.step()

        loss_value = loss.data.item()

        del loss
        torch.cuda.empty_cache()
        # train_res, loss
        return loss_value,train_res["MAPE"],train_res['MSE'],train_res["likelihood"],train_res["kl_first_p"],train_res["std_first_p"],forward_nfe,backward_nfe

    def train_epoch(epo):
        model.train()
        loss_list = []
        MAPE_list = []
        MSE_list = []
        likelihood_list = []
        kl_first_p_list = []
        std_first_p_list = []
        forward_nfe_list = []
        backward_nfe_list = []

        torch.cuda.empty_cache()

        for itr in tqdm(range(train_batch)):

            #utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=args.lr / 10)
            wait_until_kl_inc = 1000

            if itr < wait_until_kl_inc:
                kl_coef = 1
            else:
                kl_coef = 1*(1 - 0.99 ** (itr - wait_until_kl_inc))

            batch_dict_encoder = utils.get_next_batch_new(train_encoder, device)
            batch_dict_graph = utils.get_next_batch_new(train_graph, device)
            batch_dict_decoder = utils.get_next_batch(train_decoder, device)

            loss, MAPE,MSE,likelihood,kl_first_p,std_first_p,forward_nfe,backward_nfe = train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef)

            #saving results
            loss_list.append(loss), MAPE_list.append(MAPE), MSE_list.append(MSE),likelihood_list.append(
               likelihood), forward_nfe_list.append(forward_nfe),backward_nfe_list.append(backward_nfe)
            kl_first_p_list.append(kl_first_p), std_first_p_list.append(std_first_p)

            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder
                #train_res, loss
            torch.cuda.empty_cache()

        scheduler.step()

        message_train = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | MAPE {:.6F} | RMSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f} | ->nfe {:7.2f} | <-nfe {:7.2f}'.format(
            epo,
            np.mean(loss_list), np.mean(MAPE_list),np.sqrt(np.mean(MSE_list)), np.mean(likelihood_list),
            np.mean(kl_first_p_list), np.mean(std_first_p_list), np.mean(forward_nfe_list),np.mean(backward_nfe_list))

        return message_train,kl_coef
