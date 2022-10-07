import os
import sys
import argparse

import torch
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 20)

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.torch_utils import * 
from utils.dataloader import * 
from utils.utils import *
from layers.gnn_models import *
from layers.gnn_layers import *

from utils.gnn_trainer import Trainer

parser = argparse.ArgumentParser()

# Data path
parser.add_argument('--data_type', type= str, default= 'gestures', 
                    help= 'one of: gestures, elec, wind')
parser.add_argument('--data_path', type= str, default= './data/gesture')
parser.add_argument('--tr', type= float, default= 0.7, 
                help= 'the ratio of training data to the original data')
parser.add_argument('--val', type= float, default= 0.2, 
                help= 'the ratio of validation data to the original data')
parser.add_argument('--standardize', action= 'store_true', 
                help= 'standardize the inputs if it is true.')
parser.add_argument('--prob', type= float, default= 0.1, 
                help= 'the ratio of missing data to make in the original data')
parser.add_argument('--test_missing_prob', type= float, default= 0.1, 
                help= 'the ratio of missing data to make in the original data')
parser.add_argument('--test_all_missing', action= 'store_true', 
                help= 'the number of missing values to generate by row. (depreciated, it is auto-set)')
parser.add_argument('--test_n_missing', type= int, default= 1, 
                help= 'the number of missing values to generate by row. (default= 1)')
# parser.add_argument("--cat_features", nargs="+", default= None, 
#                 help= 'the indices of categorical features (list type, default= None)')

# Training options
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--patience', type=int, default=30, help='patience of early stopping condition')
parser.add_argument('--delta', type= float, default=0., help='significant improvement to update a model')
parser.add_argument('--print_log_option', type= int, default= 10, help= 'print training loss every print_log_option')
parser.add_argument('--imp_loss_penalty', type= float, default= 1., 
                    help= 'the penalty term of imputation loss')
parser.add_argument('--T_max', type= int, default= 5, help= 'T_max of consine annealing scheduler')

# model options
parser.add_argument('--model_path', type= str, default= './data/gesture/model',
                    help= 'a path to (save) the model')
parser.add_argument('--auto_set_emb_size', action= 'store_true', help= 'auto set the embedding sizes')
parser.add_argument('--num_layers', type= int, default= 4, 
                    help= 'the number of gcn layers')
parser.add_argument('--node_emb_size', type= int, default= 64,
                    help= 'the size of node embedding')
parser.add_argument('--edge_emb_size', type= int, default= 64,
                    help= 'the size of edge embedding')
parser.add_argument('--msg_emb_size', type= int, default= 64,
                    help= 'the size of message embedding')
parser.add_argument('--edge_drop_p', type= float, default= 0.3,
                    help= 'dropout ratio (default= 0.3)')
parser.add_argument('--residual_stack', action= 'store_true', default= False)

# To test
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--model_file', type= str, default= 'latest_checkpoint.pth.tar'
                    ,help= 'model file', required= False)
parser.add_argument('--model_name', type= str, default= 'latest_checkpoint.pth.tar'
                    ,help= 'model name')
parser.add_argument('--model_type', type= str, default= 'grape', 
                    help= 'grape: GRAPE, ')

parser.add_argument('--num_folds', type= int, default= 1, 
                    help = 'the number of folds')
parser.add_argument('--test_results_path', type= str, default= './test_results', 
                    help= 'a path to save the results')

args = parser.parse_args()
print(args)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.model_path): 
    print('Making a path to save the model...')
    os.makedirs(args.model_path, exist_ok= True)
else:
    print("The path already exists, skip making the path...")

# make a path to save a model 
if not os.path.exists(args.test_results_path):
    print("Making a path to save the results...")
    os.makedirs(args.test_results_path, exist_ok= True)
else:
    print("The path already exists, skip making the path...")
    
def main(args): 

    # load data
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde, task_type \
        = load_data(args)
    # define training, validation, test datasets and their dataloaders respectively 
    train_data, valid_data, test_data \
        = BipartiteData(X_train_tilde, X_train, y_train),\
            BipartiteData(X_valid_tilde, X_valid, y_valid),\
            BipartiteData(X_test_tilde, X_test, y_test)
    train_loader, valid_loader, test_loader \
        = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn= collate_fn),\
            DataLoader(valid_data, batch_size = args.batch_size, shuffle = True, collate_fn= collate_fn),\
            DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn= collate_fn)    
    
    # auto setting
    if args.auto_set_emb_size:
        n = 2**int(np.ceil(np.log2(args.input_size))-1)
        args.node_emb_size = n 
        args.edge_emb_size = n
        args.msg_emb_size = n

    # model
    # input_size(=num_features), num_labels(n_labels), cat_vars_pos, numeric_vars_pos are obtaiend after loading the data
    if args.model_type == 'grape':
        model = Grape(
            args.input_size, 
            args.n_labels,
            args.cat_vars_pos,
            args.numeric_vars_pos, 
            args.num_layers, 
            args.node_emb_size,
            args.edge_emb_size, 
            args.msg_emb_size, 
            args.edge_drop_p, 
            imp_loss_penalty= args.imp_loss_penalty,
            device= device,
            task_type= args.task_type,
            residual_stack = args.residual_stack
            ).to(device)
        # args.cat_features = None
    else:
        print("The model is yet to be implemented.")
        sys.exit()    
    print('Model config')
    print(f'task type: {model.task_type}')

    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay= 0.01)    
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.T_max) 
    early_stopping = EarlyStopping(
        patience= args.patience,
        verbose= True,
        delta = args.delta,
        path= args.model_path,
        model_name= args.model_name
    ) 

    trainer = Trainer()

    if args.test: 
        print('loading the saved model')
        model_file = os.path.join(args.model_path, args.model_file)
        ckpt = torch.load(model_file)
        model.load_state_dict(ckpt['state_dict'])
        print('loading done!')
    else: 
        print('start training...')
        # trainer(args, model, train_loader, valid_loader, early_stopping, optimizer, scheduler, device)
        trainer(args, model, train_loader, valid_loader, early_stopping, optimizer, device)
    
    print("==============================================")
    print("Testing the model...")  
    print('loading the saved model')
    model_file = os.path.join(args.model_path, args.model_name)
    ckpt = torch.load(model_file)
    model.load_state_dict(ckpt['state_dict'])
    print('loading done!')   
    perfs = trainer.test(args, model, test_loader, device)
    for k, perf in perfs.items(): 
        print(f'{k}: {perf:.4f}')
        
    return perfs

if __name__ == '__main__': 
    perf = main(args)
    perfs = dict().fromkeys(perf, None)
    for k in perfs.keys():
        perfs[k] = [perf[k]]
    for i in range(1, args.num_folds): 
        perf = main(args)
        for k in perfs.keys():
            perfs[k].append(perf[k])
    perfs_df = pd.DataFrame(perfs)
    perfs_df = perfs_df.append(perfs_df.mean().to_dict(), ignore_index= True)
    perfs_df = perfs_df.append(perfs_df.std().to_dict(), ignore_index= True)
    perfs_df.index = [str(i) for i in range(len(perfs_df)-2)] + ['mean', 'std']

    perfs_path = os.path.join(args.test_results_path, f'{args.model_type}/{args.data_type}')
    os.makedirs(perfs_path, exist_ok= True)
    pefs_df_file = os.path.join(perfs_path, f'{args.model_type}_missing_{args.test_n_missing}_{args.test_missing_prob}.csv')
    perfs_df.to_csv(pefs_df_file)
    print(perfs_df)
