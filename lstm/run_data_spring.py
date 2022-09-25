import os
import sys
sys.path.append("/home1/yjy/CG-ODE")
import time
import random
from lib.load_data_spring import ParseData
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import lib.utils as utils
from lstm.load_data_spring import ParseData
from create_lstm_model import create_LSTM_model



parser = argparse.ArgumentParser('LSTM')

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('--dataset', type=str, default='spring', help="Dec")
parser.add_argument('--datapath', type=str, default='data/', help="default data path")
parser.add_argument('--pred_length', type=int, default=48, help="Time to predict ")
parser.add_argument('--condition_length', type=int, default=71, help="Time condition on")
parser.add_argument('--features', type=str,
                    default="X,Y,vel_X,vel_Y",
                    help="selected features")
# parser.add_argument(a'--split_interval', type=int, default=3,
#                     help="number of days between two adjacent starting date of two series.")
parser.add_argument('--feature_out', type=str, default='X,Y',
                    help="X,Y")
parser.add_argument('--test_K', type=int, default=30, help="Sample numuber for testing")

parser.add_argument('-g','--GRU',action='store_true')

parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-3, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--edge_lamda', type=float, default=0.5, help='edge weight')

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
if args.feature_out == "X,Y":
    args.output_dim = 2
    args.feature_out_index = [0, 1]
else:
    raise NotImplementedError


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
    train_encoder, train_decoder, train_batch, num_atoms = dataloader.load_train_data(is_train=True)
    val_encoder, val_decoder, val_batch, _ = dataloader.load_train_data(is_train=False)
    args.num_atoms = num_atoms
    input_dim = dataloader.num_features
    output_dim = 2

    # Model Setup
    # Create the model
    model = create_LSTM_model(args,input_dim,output_dim,device)

    # Training Setup
    experimentID = time.strftime("%m-%d_%H:%M", time.localtime(time.time()+8*60**2))
    log_path = "baseline_logs/" + args.alias +"_" + args.dataset +  "_spring_"  + str(args.condition_length) +  ("_GRU_" if args.GRU else "_LSTM_") + str(args.pred_length) + "_" + str(experimentID) + ".log"
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


    def train_single_batch(model,batch_dict_encoder,batch_dict_decoder,kl_coef=None):

        optimizer.zero_grad()
        train_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder,args.num_atoms,edge_lamda = args.edge_lamda, kl_coef=kl_coef,istest=False)
        loss = train_res["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        loss_value = loss.data.item()

        del loss
        torch.cuda.empty_cache()
        # train_res, loss
        return loss_value,train_res["MAPE"],train_res['MSE'],train_res['MAE'],train_res["likelihood"],train_res["kl_first_p"],train_res["std_first_p"], 0,0


    def train_epoch(epo):
        model.train()
        loss_list = []
        MAPE_list = []
        MSE_list = []
        MAE_list = []
        likelihood_list = []
        kl_first_p_list = []
        std_first_p_list = []
        forward_nfe_list = []
        backward_nfe_list = []

        torch.cuda.empty_cache()

        for itr in tqdm(range(train_batch)):

            batch_dict_encoder = utils.get_next_batch_new(train_encoder, device)
            batch_dict_decoder = utils.get_next_batch(train_decoder, device)

            loss, MAPE,MSE,MAE,likelihood,kl_first_p,std_first_p,forward_nfe,backward_nfe = train_single_batch(model,batch_dict_encoder,batch_dict_decoder)

            #saving results
            loss_list.append(loss), MAPE_list.append(MAPE), MSE_list.append(MSE),likelihood_list.append(
               likelihood), forward_nfe_list.append(forward_nfe),backward_nfe_list.append(backward_nfe)
            kl_first_p_list.append(kl_first_p), std_first_p_list.append(std_first_p)

            del batch_dict_encoder, batch_dict_decoder
                #train_res, loss
            torch.cuda.empty_cache()

        scheduler.step()
        message_train = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | MAPE {:.6F} | RMSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f} | ->nfe {:7.2f} | <-nfe {:7.2f}'.format(
            epo,
            np.mean(loss_list), np.mean(MAPE_list),np.sqrt(np.mean(MSE_list)), np.mean(likelihood_list),
            np.mean(kl_first_p_list), np.mean(std_first_p_list), np.mean(forward_nfe_list),np.mean(backward_nfe_list))

        return message_train

    def val_epoch(epo,kl_coef=None):
        model.eval()
        MAPE_list = []
        MSE_list = []
        MAE_list = []



        torch.cuda.empty_cache()

        for itr in tqdm(range(val_batch)):
            batch_dict_encoder = utils.get_next_batch_new(val_encoder, device)
            batch_dict_decoder = utils.get_next_batch(val_decoder, device)

            val_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder,
                                                 args.num_atoms, edge_lamda=args.edge_lamda, kl_coef=kl_coef,
                                                 istest=False)

            MAPE_list.append(val_res['MAPE']), MSE_list.append(val_res['MSE']),MAE_list.append(val_res['MAE'])
            del batch_dict_encoder, batch_dict_decoder
            # train_res, loss
            torch.cuda.empty_cache()


        message_val = 'Epoch {:04d} [Val seq (cond on sampled tp)] |  MAPE {:.6F} | RMSE {:.6F} |'.format(
            epo,
            np.mean(MAPE_list), np.sqrt(np.mean(MSE_list)))

        return message_val, np.mean(MAPE_list), np.sqrt(np.mean(MSE_list)),np.mean(MAE_list)


    def test_data_covid(model, pred_length, condition_length, dataloader,device,args,kl_coef):


        encoder, decoder, num_batch = dataloader.load_test_data(pred_length=pred_length,
                                                                    condition_length=condition_length)


        total = {}
        total["loss"] = 0
        total["likelihood"] = 0
        total["MAPE"] = 0
        total["RMSE"] = 0
        total["MSE"] = 0
        total["MAE"] = 0
        total["kl_first_p"] = 0
        total["std_first_p"] = 0
        MAPE_each = []
        RMSE_each = []
        MAE_each=[]

        n_test_batches = 0

        model.eval()
        print("Computing loss... ")
        with torch.no_grad():
            for _ in tqdm(range(num_batch)):
                batch_dict_encoder = utils.get_next_batch_new(encoder, device)
                batch_dict_decoder = utils.get_next_batch_test(decoder, device)

                results = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, args.num_atoms,
                                                edge_lamda=args.edge_lamda, kl_coef=kl_coef, istest=True)

                for key in total.keys():
                    if key in results:
                        var = results[key]
                        if isinstance(var, torch.Tensor):
                            var = var.detach().item()
                        if key =="MAPE":
                            MAPE_each.append(var)
                        elif key == "MAE":
                            MAE_each.append(var)
                        elif key == "MSE": # assign value for both MSE and RMSE
                            RMSE_each.append(np.sqrt(var))
                            total["RMSE"] += np.sqrt(var)
                        total[key] += var

                n_test_batches += 1

                del batch_dict_encoder, batch_dict_decoder, results

            if n_test_batches > 0:
                for key, value in total.items():
                    total[key] = total[key] / n_test_batches


        return total, utils.print_MAPE(MAPE_each), utils.print_MAPE(RMSE_each), utils.print_MAPE(MAE_each)

    # Training and Testing
    for epo in range(1, args.niters +1):

        message_train = train_epoch(epo)
        message_val, MAPE_val, RMSE_val,MAE_val = val_epoch(epo)

        if epo % n_iters_to_viz == 0:
            # Logging Train and Val
            logger.info("Experiment " + str(experimentID))
            logger.info(message_train)
            logger.info(message_val)


            # Testing
            model.eval()
            test_res,MAPE_each,RMSE_each,MAE_each = test_data_covid(model, args.pred_length, args.condition_length, dataloader,
                                 device=device, args = args, kl_coef=None)
            message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | MAPE {:.6F} | RMSE {:.6F}| MAE {:.6F}|'.format(
                epo,
                test_res["MAPE"], test_res["RMSE"],test_res["MAE"])


            if MAPE_val < best_val_MAPE:
                best_val_MAPE = MAPE_val
                best_val_RMSE = RMSE_val
                logger.info("Best Val!")
                # don't save

            logger.info(message_test)
            logger.info(MAPE_each)
            logger.info(RMSE_each)


            if test_res["MAPE"] < best_test_MAPE:
                best_test_MAPE = test_res["MAPE"]
                best_test_RMSE = test_res["RMSE"]
                message_best = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Best Test MAPE {:.6f}|Best Test RMSE {:.6f}|'.format(epo,
                                                                                                        best_test_MAPE,best_test_RMSE)
                logger.info(MAPE_each)
                logger.info(RMSE_each)
                logger.info(message_best)

            torch.cuda.empty_cache()
        ''

