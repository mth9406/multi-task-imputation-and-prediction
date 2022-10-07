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

from utils.torch_utils import * 
from layers.models import *
from utils.train import *
from utils.dataloader import * 

from sklearn.utils.class_weight import compute_class_weight
from fancyimpute import SoftImpute

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
                help= 'force every observation in the test data to have missing values.')
parser.add_argument('--test_n_missing', type= int, default= 1, 
                help= 'the number of missing values to generate by row. (depreciated, it is auto-set)')
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
parser.add_argument('--input_size', type= int, default= 18, 
                    help= 'the number of time series (dimension)')
parser.add_argument('--n_labels', type= int, default= 5, 
                    help= 'the number of labels (default= 5)')
parser.add_argument('--drop_p', type= float, default= 0.5,
                    help= 'dropout ratio (default= 0.5)')
parser.add_argument('--stack_ae_lyrs', action= 'store_true', 
                    help= 'auto-encoder layer will have 2 linear layers respectively if it is set true.')
parser.add_argument('--stack_fc_lyrs', action= 'store_true',
                    help= 'fc_out layer will have 2 Linear layers if it is set true.')

# To test
parser.add_argument('--model_name', type= str, default= 'latest_checkpoint.pth.tar'
                    ,help= 'model name', required= False)
parser.add_argument('--model_type', type= str, default= 'aei', 
                    help= 'aei: AutoEcoderImpute, vai: VarationalAutoImpute')
parser.add_argument('--vai_n_samples', type= int, default= 100, 
                    help= 'sampling size of VAI model')

parser.add_argument('--num_folds', type= int, default= 1, 
                    help = 'the number of folds')
parser.add_argument('--test_results_path', type= str, default= './test_results_tp', 
                    help= 'a path to save the results of a part (imputation) of two phase model')

args = parser.parse_args()
print(args)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make a path to save a model 
if not os.path.exists(args.model_path):
    print("Making a path to save the model...")
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
    # read data
    # one of: gestures, physionet, mimic3
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde, task_type \
        = load_data(args)
    
    # find masks
    M_train, M_valid, M_test = make_mask(X_train_tilde), make_mask(X_valid_tilde), make_mask(X_test_tilde)
    X_train_tilde = torch.FloatTensor(SoftImpute(verbose= False).fit_transform(X_train_tilde))
    X_valid_tilde = torch.FloatTensor(SoftImpute(verbose= False).fit_transform(X_valid_tilde))
    X_test_tilde = torch.FloatTensor(SoftImpute(verbose= False).fit_transform(X_test_tilde))   

    # define training, validation, test datasets and their dataloaders respectively 
    train_data, valid_data, test_data \
        = TableDataset(X_train_tilde, M_train, y_train, X_comp= X_train),\
            TableDataset(X_valid_tilde, M_valid, y_valid, X_comp= X_valid),\
            TableDataset(X_test_tilde, M_test, y_test, X_comp= X_test)
    train_loader, valid_loader, test_loader \
        = DataLoader(train_data, batch_size = args.batch_size, shuffle = True),\
            DataLoader(valid_data, batch_size = args.batch_size, shuffle = True),\
            DataLoader(test_data, batch_size = args.batch_size, shuffle = False)    

    print("Loading data done!")

    # model
    if args.model_type == 'aei':
        model = AutoEncoder(args.input_size, args.drop_p, stack_fc_lyrs=args.stack_fc_lyrs, stack_ae_lyrs= args.stack_ae_lyrs).to(device)
    elif args.model_type == 'vai': 
        model=VariationalAutoEncoder(args.input_size, args.drop_p, stack_fc_lyrs=args.stack_fc_lyrs, stack_ae_lyrs= args.stack_ae_lyrs).to(device)
    else:
        print("The model is yet to be implemented.")
        sys.exit()

    # setting training args...
    if args.task_type == 'regr': 
        criterion = nn.MSELoss()
    elif args.task_type == 'cls' and args.n_labels > 1: 
        criterion = nn.CrossEntropyLoss()
    else: 
        criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), args.lr)    
    early_stopping = EarlyStopping(
        patience= args.patience,
        verbose= True,
        delta = args.delta,
        path= args.model_path,
        model_name= args.model_name
    ) 

    train(args, model, train_loader, valid_loader, optimizer, criterion, early_stopping, device)
    
    # print("==============================================")
    # print("Testing the imputation model...")
    # perf = test_imp(args, model, test_loader, criterion, device)

    # test the linear model upon the imputation model
    print("==============================================")
    print("Training the linear model after the imputation...")
    linear_model = LinearModel(args.input_size, args.n_labels, args.drop_p, args.stack_fc_lyrs).to(device)
    
    model.eval()
    inputs_train = {'input':X_train_tilde.to(device), 'mask':M_train.to(device)}
    inputs_valid = {'input':X_valid_tilde.to(device), 'mask':M_valid.to(device)}
    inputs_test = {'input':X_test_tilde.to(device), 'mask':M_test.to(device)}
    with torch.no_grad():
        X_train_tilde = model.forward(inputs_train)['imputation'].detach().cpu()
        X_valid_tilde = model.forward(inputs_valid)['imputation'].detach().cpu()
        X_test_tilde = model.forward(inputs_test)['imputation'].detach().cpu()

    assert torch.isnan(X_train_tilde).sum() == 0, 'Imputation failed'
    assert torch.isnan(X_valid_tilde).sum() == 0, 'Imputation failed'
    assert torch.isnan(X_test_tilde).sum() == 0, 'Imputation failed'
    
    # define training, validation, test datasets and their dataloaders respectively 
    train_data, valid_data, test_data \
        = TableDataset(X_train_tilde, M_train, y_train, X_comp= X_train),\
            TableDataset(X_valid_tilde, M_valid, y_valid, X_comp= X_valid),\
            TableDataset(X_test_tilde, M_test, y_test, X_comp= X_test)
    train_loader, valid_loader, test_loader \
        = DataLoader(train_data, batch_size = args.batch_size, shuffle = True),\
            DataLoader(valid_data, batch_size = args.batch_size, shuffle = True),\
            DataLoader(test_data, batch_size = args.batch_size, shuffle = False)

    optimizer = optim.Adam(linear_model.parameters(), args.lr)    
    early_stopping = EarlyStopping(
        patience= args.patience,
        verbose= True,
        delta = args.delta,
        path= args.model_path
    ) 

    train(args, linear_model, train_loader, valid_loader, optimizer, criterion, early_stopping, device)
    
    print("==============================================")
    print("Testing the linear model...")
    print('loading the saved model')
    model_file = os.path.join(args.model_path, args.model_name)
    ckpt = torch.load(model_file)
    linear_model.load_state_dict(ckpt['state_dict'])
    print('loading done!')  
    if task_type == 'cls': 
        perf_task = test_cls(args, linear_model, test_loader, criterion, device)
    else: 
        perf_task = test_regr(args, linear_model, test_loader, criterion, device)
   
    return perf_task 

if __name__ == '__main__': 

    if args.num_folds == 1:
        main(args)
    else: 
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

        perfs_path = os.path.join(args.test_results_path, f'{args.data_type}')
        os.makedirs(perfs_path, exist_ok= True)
        pefs_df_file = os.path.join(perfs_path, f'{args.model_type}_missing_{args.test_n_missing}_{args.test_missing_prob}.csv')
        perfs_df.to_csv(pefs_df_file)

        for k, v in perfs.items():
            perfs[k] = [np.mean(perfs[k]), np.std(perfs[k])]
        
        print("==============================================")
        print(f"Model type: {args.model_type}")
        print(f"Data: {args.data_type}")
        print(f"Column size: {args.input_size}")
        print(f"Number of classes: {args.n_labels}") 
        if args.test_all_missing:
            print(f"The number of missing values per row: {args.test_n_missing}")
        else:
            print(f"Missing rate: {args.prob}")
        print("==============================================")

        for k, v in perfs.items(): 
            print(f"{k}: mean= {v[0]:.3f}, std= {v[1]:.3f}     ")