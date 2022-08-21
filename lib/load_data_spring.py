from asyncio import all_tasks
import numpy as np
import torch
from torch_geometric.data import DataLoader,Data
from torch.utils.data import DataLoader as Loader
from tqdm import tqdm
import math
from scipy.linalg import block_diag
import lib.utils as utils
import pandas as pd
from einops import repeat
from torch.nn.utils.rnn import pad_sequence
from warnings import warn


weights = [ 0.1]#-1位置是图需要除的一个数


class ParseData(object):

    def __init__(self,args):
        self.args = args
        self.datapath = args.datapath
        self.dataset = args.dataset
        self.random_seed = args.random_seed
        self.pred_length = args.pred_length
        self.condition_length = args.condition_length
        self.batch_size = args.batch_size

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

    def load_train_data(self,is_train = True):

        # Loading Data. N is state number, T is number of days. D is feature number.

        locs = np.load(self.args.datapath + self.args.dataset + '/loc_train_springs5.npy',allow_pickle=True) # [K,N,T,2]
        vels = np.load(self.args.datapath + self.args.dataset + '/vel_train_springs5.npy',allow_pickle=True) # [K,N,T,2]

        graphs = np.load(self.args.datapath + self.args.dataset + '/edges_train_springs5.npy')  # [K,N,N]

        # Normalize features to [-1, 1]
        locs, self.max_loc, self.min_loc = self.normalize_features(locs,locs.shape[-1])  # [num_sims,num_atoms, (timestamps,2)]
        vels, self.max_vel, self.min_vel = self.normalize_features(vels,vels.shape[-1] )
        
        # Graph Preprocessing: remain self-loop and take log
        graphs = self.graph_preprocessing(graphs,method = 'norm_const', is_self_loop = True)  #[K,N,N]

        features = np.concatenate((locs,vels),axis=-1) # [K,N,T,D=4]
        graphs = repeat(graphs, 'K N n -> K T N n', T=features.shape[-2]) # [K,T,N,N]

        self.num_states = features.shape[1]
        self.num_features = features.shape[-1]
        if is_train:
            features = features[:-5, :, :, :]
            graphs = graphs[:-5, :, :, :]
        else:
            features = features[-5:, :, :, :]
            graphs = graphs[-5:, :, :, :]
        
        encoder_data_loader, decoder_data_loader, decoder_graph_loader, num_batch, self.num_states = self.generate_train_val_dataloader(features,graphs,is_train)


        return encoder_data_loader, decoder_data_loader, decoder_graph_loader, num_batch, self.num_states
    
    def generate_train_val_dataloader(self,features,graphs,is_train = True, is_test=False):
        # Split data for encoder and decoder dataloader
        feature_observed, times_observed, series_decoder, times_extrap = self.split_data(
            features)  # series_decoder[K*N,T2,D]
        self.times_extrap = times_extrap

        # Generate Encoder data
        encoder_data_loader = self.transfer_data(feature_observed, graphs, times_observed, self.batch_size)

        # Generate Decoder Data and Graph
        if is_train:
            series_decoder_gt = self.decoder_gt_train()  # [K*N,T2,D]
        else:
            series_decoder_gt = self.decoder_gt_train()  # [K*N,T2,D]
            series_decoder_gt = series_decoder_gt[-5*self.num_states:,:,:]

        series_decoder_all = [(series_decoder[i, :, :], series_decoder_gt[i, :, :]) for i in
                              range(series_decoder.shape[0])]
        if not is_test:
            decoder_data_loader = Loader(series_decoder_all, batch_size=self.batch_size * self.num_states, shuffle=False,
                                        collate_fn=lambda batch: self.variable_time_collate_fn_activity(
                                            batch))  # num_graph*num_ball [tt,vals,masks]
        else:
            decoder_data_loader = Loader(series_decoder_all, batch_size=self.batch_size * self.num_states, shuffle=False,
                                        collate_fn=lambda batch: self.variable_test(batch))  #

        graph_decoder = graphs[:, self.args.condition_length:, :, :]  # [K,T2,N,N]
        decoder_graph_loader = Loader(graph_decoder, batch_size=self.batch_size, shuffle=False)

        num_batch = len(decoder_data_loader)
        assert len(decoder_data_loader) == len(decoder_graph_loader)

        # Inf-Generator
        encoder_data_loader = utils.inf_generator(encoder_data_loader)
        decoder_graph_loader = utils.inf_generator(decoder_graph_loader)
        decoder_data_loader = utils.inf_generator(decoder_data_loader)

        return encoder_data_loader, decoder_data_loader, decoder_graph_loader, num_batch, self.num_states

    def transfer_data(self, feature, edges, times,batch_size):
        '''
        :param feature: #[K,N,T1,D]
        :param edges: #[K,T,N,N], with self-loop
        :param times: #[T1]
        :param time_begin: 1
        :return:
        '''
        data_list = []
        edge_size_list = []

        num_samples = feature.shape[0]

        for i in tqdm(range(num_samples)):
            data_per_graph, edge_size = self.transfer_one_graph(feature[i], edges[i], times)
            data_list.append(data_per_graph)
            edge_size_list.append(edge_size)

        print("average number of edges per graph is %.4f" % np.mean(np.asarray(edge_size_list)))
        data_loader = DataLoader(data_list, batch_size=batch_size,shuffle=False)

        return data_loader

    def transfer_one_graph(self,feature, edge, time):
        '''f

        :param feature: [N,T1,D]
        :param edge: [T,N,N]  (needs to transfer into [T1,N,N] first, already with self-loop)
        :param time: [T1]
        :return:
            1. x : [N*T1,D]: feature for each node.
            2. edge_index [2,num_edge]: edges including cross-time
            3. edge_weight [num_edge]: edge weights
            4. y: [N], value= num_steps: number of timestamps for each state node.
            5. x_pos 【N*T1】: timestamp for each node
            6. edge_time [num_edge]: edge relative time.
        '''

        ########## Getting and setting hyperparameters:
        num_states = feature.shape[0]
        T1 = self.args.condition_length
        each_gap = 1/ edge.shape[0]
        edge = edge[:T1,:,:]
        time = np.reshape(time,(-1,1))

        ########## Compute Node related data:  x,y,x_pos
        # [Num_states],value is the number of timestamp for each state in the encoder, == args.condition_length
        y = self.args.condition_length*np.ones(num_states)
        # [Num_states*T1,D]
        x = np.reshape(feature,(-1,feature.shape[2]))
        # [Num_states*T1,1] node timestamp
        x_pos = np.concatenate([time for i in range(num_states)],axis=0)
        assert len(x_pos) == feature.shape[0]*feature.shape[1]

        ########## Compute edge related data
        edge_time_matrix = np.concatenate([np.asarray(x_pos).reshape(-1, 1) for _ in range(len(x_pos))],
                                          axis=1) - np.concatenate(
            [np.asarray(x_pos).reshape(1, -1) for _ in range(len(x_pos))], axis=0)  # [N*T1,N*T1], SAME TIME = 0

        edge_exist_matrix = np.ones((len(x_pos), len(x_pos)))  # [N*T1,N*T1] NO-EDGE = 0, depends on both edge weight and time matrix

        # Step1: Construct edge_weight_matrix [N*T1,N*T1]
        edge_repeat = np.repeat(edge, self.args.condition_length, axis=2)  # [T1,N,NT1]
        edge_repeat = np.transpose(edge_repeat, (1, 0, 2))  # [N,T1,NT1]
        edge_weight_matrix = np.reshape(edge_repeat, (-1, edge_repeat.shape[2]))  # [N*T1,N*T1]

        # mask out cross_time edges of different state nodes.
        a = np.identity(T1)  # [T,T]
        b = np.concatenate([a for i in range(num_states)], axis=0)  # [N*T,T]
        c = np.concatenate([b for i in range(num_states)], axis=1)  # [N*T,N*T]

        a = np.ones((T1, T1))
        d = block_diag(*([a] * num_states))
        edge_weight_mask = (1 - d) * c + d
        edge_weight_matrix = edge_weight_matrix * edge_weight_mask  # [N*T1,N*T1]

        max_gap = each_gap


        # Step2: Construct edge_exist_matrix [N*T1,N*T1]: depending on both time and weight.
        edge_exist_matrix = np.where(
            (edge_time_matrix <= 0) & (abs(edge_time_matrix) <= max_gap) & (edge_weight_matrix != 0),
            edge_exist_matrix, 0)


        edge_weight_matrix = edge_weight_matrix * edge_exist_matrix
        edge_index, edge_weight_attr = utils.convert_sparse(edge_weight_matrix)
        if np.sum(edge_weight_matrix!=0)==0:
            warn('No edge in one graph') #at least one edge weight (one edge) exists.

        edge_time_matrix = (edge_time_matrix + 3) * edge_exist_matrix # padding 2 to avoid equal time been seen as not exists.
        _, edge_time_attr = utils.convert_sparse(edge_time_matrix)
        edge_time_attr -= 3

        # converting to tensor
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_weight_attr = torch.FloatTensor(edge_weight_attr)
        edge_time_attr = torch.FloatTensor(edge_time_attr)
        y = torch.LongTensor(y)
        x_pos = torch.FloatTensor(x_pos)


        graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight_attr, y=y, pos=x_pos, edge_time = edge_time_attr)
        edge_num = edge_index.shape[1]

        return graph_data,edge_num

    def load_test_data(self,pred_length,condition_length):

        # Loading Data. N is state number, T is number of days. D is feature number.

        locs = np.load(self.args.datapath + self.args.dataset + '/loc_test_springs5.npy',allow_pickle=True) # [K,N,T,2]
        vels = np.load(self.args.datapath + self.args.dataset + '/vel_test_springs5.npy',allow_pickle=True) # [K,N,T,2]

        graphs = np.load(self.args.datapath + self.args.dataset + '/edges_test_springs5.npy')  # [K,N,N]

        # Normalize features to [-1, 1]
        locs, self.max_loc, self.min_loc = self.normalize_features(locs,locs.shape[-1])  # [num_sims,num_atoms, (timestamps,2)]
        vels, self.max_vel, self.min_vel = self.normalize_features(vels,vels.shape[-1])
        
        # Graph Preprocessing: remain self-loop and take log
        graphs = self.graph_preprocessing(graphs,method = 'norm_const', is_self_loop = True)  #[K,N,N]

        features = np.concatenate((locs,vels),axis=-1) # [K,N,T,D=4]
        graphs = repeat(graphs, 'K N n -> K T N n', T=features.shape[-2]) # [K,T,N,N]
        
        encoder_data_loader, decoder_data_loader, decoder_graph_loader, num_batch, self.num_states = self.generate_train_val_dataloader(features,graphs,is_test=True)

        return encoder_data_loader, decoder_data_loader, decoder_graph_loader, num_batch

    def split_data(self, feature):
        '''
               Generate encoder data (need further preprocess) and decoder data
               :param feature: [K,N,T,D], T=T1+T2
               :param data_type:
               :return:
               '''
        feature_observed = feature[:, :, :self.args.condition_length, :]
        # select corresponding features
        feature_out_index = self.args.feature_out_index
        feature_extrap = feature[:, :, self.args.condition_length:, feature_out_index]
        assert feature_extrap.shape[-1] == len(feature_out_index)
        times = np.asarray([i / feature.shape[2] for i in range(feature.shape[2])])  # normalized in [0,1] T
        times_observed = times[:self.args.condition_length]  # [T1]
        times_extrap = times[self.args.condition_length:] - times[
            self.args.condition_length]  # [T2] making starting time of T2 be 0.
        assert times_extrap[0] == 0
        series_decoder = np.reshape(feature_extrap, (-1, len(times_extrap), len(feature_out_index)))  # [K*N,T2,D]

        return feature_observed, times_observed, series_decoder, times_extrap

        ''
    def graph_preprocessing(self,graph_input, method = 'norm_const', is_self_loop = True):
        '''
                :param graph_input: [T,N,N]
                :param method: norm--norm by rows: G[i,j] is the outflow from i to j.
                :param is_self_loop: True to remain self-loop, otherwise no self-loop
                :return: [T,N,N]
                '''
        if not is_self_loop:
            num_days = graph_input.shape[0]
            num_states = graph_input.shape[1]
            graph_output = np.ones_like(graph_input)
            for i in range(num_days):
                graph_output[i] = graph_input[i] * (1 - np.identity(num_states))
        else:
            graph_output = graph_input

        if method == "log":
            graph_output = np.log(graph_output + 1)  # 0 remains 0
        elif method == "norm_const":
            graph_output = graph_output/weights[-1]

        return graph_output

    def normalize_features(self,inputs, num_balls,is_train=True):
        '''

        :param inputs: [num-train, num-ball,(timestamps,2)]
        :return:
        '''
        # value_list_length = [balls[i].shape[0] for i in range(num_balls) for balls in inputs]  # [2500] num_train * num_ball
        # self.timelength = max(value_list_length)
        # value_list = [torch.tensor(balls[i]) for i in range(num_balls) for balls in inputs]
        # value_padding = pad_sequence(value_list,batch_first=True,padding_value = 0)
        # max_value = torch.max(value_padding).item()
        # min_value = torch.min(value_padding).item()
        max_value = inputs.max() if not hasattr(self, 'max_value') else self.max_value
        min_value = inputs.min() if not hasattr(self, 'min_value') else self.min_value # 当输入维度全同时非常简单

        # Normalize to [-1, 1]
        inputs = (inputs - min_value) * 2 / (max_value - min_value) - 1
        return inputs,max_value,min_value

    def decoder_gt_train(self):

        # Loading Data. N is state number, T is number of days. D is feature number.

        locs = np.load(self.args.datapath + self.args.dataset + '/loc_train_springs5.npy',allow_pickle=True) # [K,N,T,2]
        vels = np.load(self.args.datapath + self.args.dataset + '/vel_train_springs5.npy',allow_pickle=True) # [K,N,T,2]

        graphs = np.load(self.args.datapath + self.args.dataset + '/edges_train_springs5.npy')  # [K,N,N]

        # Normalize features to [-1, 1]
        locs, _, _ = self.normalize_features(locs,locs.shape[-1])  # [num_sims,num_atoms, (timestamps,2)]
        vels, _, _ = self.normalize_features(vels,vels.shape[-1])
        
        # Graph Preprocessing: remain self-loop and take log
        graphs = self.graph_preprocessing(graphs,method = 'norm_const', is_self_loop = True)  #[K,N,N]

        features = np.concatenate((locs,vels),axis=-1) # [K,N,T,D=4]
        graphs = repeat(graphs, 'K N n -> K T N n', T=features.shape[-2]) # [K,T,N,N]
        # Split data for encoder and decoder dataloader
        _, _, series_decoder, _ = self.split_data(features)  # series_decoder[K*N,T2,D]
        return series_decoder

    def variable_time_collate_fn_activity(self,batch):
        """
        Expects a batch of
            - (feature0,feaure_gt) [K*N, T2, D]
        """
        # Extract corrsponding deaths or cases
        combined_vals = np.concatenate([np.expand_dims(ex[0],axis=0) for ex in batch],axis=0)
        combined_vals_true = np.concatenate([np.expand_dims(ex[1],axis=0) for ex in batch], axis = 0)



        combined_vals = torch.FloatTensor(combined_vals) #[M,T2,D]
        combined_vals_true = torch.FloatTensor(combined_vals_true)  # [M,T2,D]

        combined_tt = torch.FloatTensor(self.times_extrap)

        data_dict = {
            "data": combined_vals,
            "time_steps": combined_tt,
            "data_gt" : combined_vals_true
            }
        return data_dict

    def variable_test(self,batch):
        """
        Expects a batch of
            - (feature,feature_gt,mask)
            - feature: [N,T,D]
            - mask: T #因为改了的缘故，没有这个mask了
        Returns:
            combined_tt: The union of all time observations. [T2], varies from different testing sample
            combined_vals: (M, T2, D) tensor containing the gt values.
            combined_masks: index for output timestamps. Only for masking out prediction.
        """
        # Extract corrsponding deaths or cases
        # Extract corrsponding deaths or cases
        combined_vals = np.concatenate([np.expand_dims(ex[0],axis=0) for ex in batch],axis=0)
        combined_vals_true = np.concatenate([np.expand_dims(ex[1],axis=0) for ex in batch], axis = 0)

        combined_vals = torch.FloatTensor(combined_vals) #[M,T2,D]
        combined_vals_true = torch.FloatTensor(combined_vals_true)  # [M,T2,D]

        combined_masks = None #[1]
        combined_tt = torch.FloatTensor(self.times_extrap)

        data_dict = {
            "data": combined_vals,
            "time_steps": combined_tt,
            "masks":combined_masks,
            "data_gt" : combined_vals_true
            }
        return data_dict
