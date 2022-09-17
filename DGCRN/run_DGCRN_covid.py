import os
import sys
sys.path.append("/nvme/yuanjingyang/CG-ODE")
sys.path.append("/nvme/yuanjingyang/CG-ODE/DGCRN/model")
import time
import random
from load_data_covid import ParseData
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
from einops import rearrange
import lib.utils as utils
from torch.distributions.normal import Normal

from model.net import DGCRN
from model.trainer import Trainer
# from lib.utils import test_data_covid


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
parser.add_argument('-b', '--batch-size', type=int, default=1)
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


parser.add_argument('--augment_dim', type=int, default=0, help='augmented dimension')
parser.add_argument('--solver', type=str, default="rk4", help='dopri5,rk4,euler')
parser.add_argument('--rtol', type=float,default=1e-2, help='tolerance of ode')
parser.add_argument('--atol', type=float,default=1e-2, help='tolerance of ode adjent')

parser.add_argument('--alias', type=str, default="DGCRN")
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
    experimentID = time.strftime("%m-%d_%H:%M", time.localtime(time.time()+8*60**2))

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
    model = DGCRN(gcn_depth = 2,
                  num_nodes = num_atoms,
                  device = device,
                  predefined_A=None, # TODO:修改
                  dropout=args.dropout,
                  subgraph_size=20,
                  node_dim=num_atoms,
                  middle_dim=2,
                  seq_length=args.condition_length,
                  in_dim=input_dim,# 修改了
                #   out_dim=args.output_dim,# 修改了
                  out_dim=5,# 修改了
                  layers=3,
                  list_weight=[0.05, 0.95, 0.95],
                  tanhalpha=3,
                  cl_decay_steps=2000,
                  rnn_size=64,
                  hyperGNN_dim=16)


    engine = Trainer(model, args.lr, args.l2, clip=5,
                     step_size = 2500, seq_out_len=12, scaler=None, device=device,
                     cl=True, new_training_method=False)


    # Training Setup
    log_path = "baseline_logs/" + args.alias +"_" + args.dataset +  "_Con_"  + str(args.condition_length) +  "_Pre_" + str(args.pred_length) + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    logger.info(args.alias)
    
    batches_seen = 0

    def train_epoch(epo):
        train_loss = []
        train_mape = []
        train_rmse = []
        train_mae = []
        
        for itr in tqdm(range(train_batch)):
            batches_seen =0 # TODO
            batch_dict_encoder = utils.get_next_batch_new(train_encoder, device) # [batch, N, T, D] 
            batch_dict_graph = utils.get_next_batch_new(train_graph, device)     # [batch, T, N, N] 
            batch_dict_decoder = utils.get_next_batch(train_decoder, device)

            batch_dict_encoder = rearrange(batch_dict_encoder, 'K N T D -> K D N T')
            metrics = engine.train(input = batch_dict_encoder,
                                graph = batch_dict_graph,
                                real_val=batch_dict_decoder['data'][:,-1,0],
                                idx=None,
                                batches_seen=batches_seen)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_mae.append(metrics[3])
            

        return train_loss, train_mape, train_rmse, train_mae

    def val_epoch(epo):
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        valid_mae = []
        for itr in tqdm(range(val_batch)):
            batches_seen =0 # TODO
            batch_dict_encoder = utils.get_next_batch_new(val_encoder, device)# [batch, N, T, D] 
            batch_dict_graph = utils.get_next_batch_new(val_graph, device)     # [batch, T, N, N] 
            batch_dict_decoder = utils.get_next_batch(val_decoder, device)

            batch_dict_encoder = rearrange(batch_dict_encoder, 'K N T D -> K D N T')
            metrics = engine.eval(input = batch_dict_encoder,
                                graph = batch_dict_graph,
                                real_val=batch_dict_decoder['data'][:,-1,0],ycl=None)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_mae.append(metrics[3])
        return valid_loss, valid_mape, valid_rmse, valid_mae

    # Training and Testing
    his_loss = []

    for epo in range(1, args.niters + 1):
        train_loss, train_mape, train_rmse, train_mae = train_epoch(epo)
        valid_loss, valid_mape, valid_rmse, valid_mae = val_epoch(epo)
        
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_mae = np.mean(train_mae)
        

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_mae = np.mean(valid_mae)
        his_loss.append(mvalid_loss)
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train MAE: {:.4f}\
                            , Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid MAE: {:.4f}'.format(
                            epo, mtrain_loss, mtrain_mape, mtrain_rmse, mtrain_mae,
                            mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_mae)
        print(log)
        logger.info(log)


    print("Training finished, begin testing")
    encoder, decoder, graph, num_batch = dataloader.load_test_data(pred_length=args.pred_length,
	 															   condition_length=args.condition_length)
    MAPE_each = []
    RMSE_each = []
    MAE_each = []
    with torch.no_grad():
        for _ in tqdm(range(num_batch)):
            batch_dict_encoder = utils.get_next_batch_new(encoder, device)
            batch_dict_graph = utils.get_next_batch_new(graph, device)
            batch_dict_decoder = utils.get_next_batch_test(decoder, device)
            
            batch_dict_encoder = rearrange(batch_dict_encoder, 'K N T D -> K D N T')
            metrics = engine.eval(input = batch_dict_encoder.float(),
                                graph = batch_dict_graph.float(),
                                real_val=batch_dict_decoder['data'][:,-1,0].float(),ycl=None)
            # valid_loss.append(metrics[0])
            MAPE_each.append(metrics[1])
            RMSE_each.append(metrics[2])
            MAE_each.append(metrics[3])
    
    log = 'TEST: Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test MAE: {:.4f}/epoch'.format(
                          np.array(MAPE_each).mean(), np.array(RMSE_each).mean(),np.array(MAE_each).mean())
    print(log)
    logger.info(log)