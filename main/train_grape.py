import os
import sys
import argparse

import torch
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.torchUtils import * 
from utils.dataloader import * 
from utils.utils import *
from layers.gnnModels import *
from layers.gnnLayers import *

from utils.gnnTrainer import Trainer

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
parser.add_argument('--test_all_missing', action= 'store_true', 
                help= 'force every observation in the test data to have missing values.')
parser.add_argument('--test_n_missing', type= int, default= 1, 
                help= 'the number of missing values to generate by row. (default= 1)')
parser.add_argument("--cat_features", nargs="+", default= None, 
                help= 'the indices of categorical features (list type, default= None)')

# Training options
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--patience', type=int, default=30, help='patience of early stopping condition')
parser.add_argument('--delta', type= float, default=0., help='significant improvement to update a model')
parser.add_argument('--print_log_option', type= int, default= 10, help= 'print training loss every print_log_option')
parser.add_argument('--imp_loss_penalty', type= float, default= 1.0, 
                    help= 'the penalty term of imputation loss')

# model options
parser.add_argument('--model_path', type= str, default= './data/gesture/model',
                    help= 'a path to (save) the model')
parser.add_argument('--drop_p', type= float, default= 0.5,
                    help= 'dropout ratio (default= 0.5)')

# To test
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--model_file', type= str, default= 'latest_checkpoint.pth.tar'
                    ,help= 'model file', required= False)
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

def main(args): 

    # model
    if args.model_type == 'grape':
        model = Grape(args.input_size, args.n_labels, args.drop_p, stack_fc_lyrs=args.stack_fc_lyrs).to(device)
        # args.cat_features = None
    else:
        print("The model is yet to be implemented.")
        sys.exit()    

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

    optimizer = optim.Adam(model.parameters(), args.lr)    
    early_stopping = EarlyStopping(
        patience= args.patience,
        verbose= True,
        delta = args.delta,
        path= args.model_path
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
        trainer(args, model, train_loader, valid_loader, early_stopping, optimizer, device)
    
    print("==============================================")
    print("Testing the model...")   
    perfs = trainer.test(model, test_loader, device)
    for k, perf in perfs.items(): 
        print(f'{k}: {perf}')
        
    return perfs