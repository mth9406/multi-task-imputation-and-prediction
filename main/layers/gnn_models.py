import torch 
from torch import nn
from torch.nn import functional as F
from layers.gnn_layers import *

import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from utils.utils import evaluate

from torch_geometric.utils import dropout_adj

# Grape 
class Grape(nn.Module):
    def __init__(self, 
                num_features,
                num_labels,
                cat_vars_pos:list= [], 
                numeric_vars_pos:list= [],
                num_layers:int= 3, 
                node_emb_size:int = 64,
                edge_emb_size:int = 64,
                msg_emb_size:int = 64,
                edge_drop_p:float = 0.3,
                imp_loss_penalty:float = 0.01,
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                ):
        super().__init__() 
        
        self.init = Init(num_features, device= device)
        self.gcn_block0 = GCNBlock(num_features, node_emb_size, 1, edge_emb_size, msg_emb_size)
        for i in range(1, num_layers): 
            setattr(self, f'gcn_block{i}', GCNBlock(node_emb_size, node_emb_size, edge_emb_size, edge_emb_size, msg_emb_size))
        self.eph = EdgePredictionHead(node_emb_size, num_features, device)
        self.nph = NodePredictionHead(num_features, num_labels)

        self.num_layers = num_layers
        self.edge_drop_p = edge_drop_p
        self.node_emb_size = node_emb_size
        self.edge_emb_size = edge_emb_size
        self.msg_emb_size = msg_emb_size
        self.num_features = num_features
        self.num_labels = num_labels 
        self.cat_vars_pos = cat_vars_pos
        self.numeric_vars_pos = numeric_vars_pos
        self.imp_loss_penalty = imp_loss_penalty
        self.device = device

    def forward(self, x):
        if self.training:
            edge_index, edge_emb = dropout_adj(x['edge_index'], x['edge_value'])
        else: 
            edge_index, edge_emb = x['edge_index'], x['edge_value']
        edge_emb = edge_emb[:, None]
        node_emb, feature_emb = self.init(x['x'].shape[0])

        node_emb, edge_emb, feature_emb = self.gcn_block0(node_emb, edge_emb, feature_emb, edge_index)
        node_emb, edge_emb, feature_emb = torch.relu(node_emb), torch.relu(edge_emb), torch.relu(feature_emb)

        for i in range(1, self.num_layers):
            node_emb_b4, edge_emb_b4, feature_emb_b4 = node_emb.clone().detach(), edge_emb.clone().detach(), feature_emb.clone().detach()
            node_emb, edge_emb, feature_emb = getattr(self, f'gcn_block{i}')(node_emb, edge_emb, feature_emb, edge_index)
            node_emb, edge_emb, feature_emb \
                = torch.relu(node_emb+node_emb_b4), torch.relu(edge_emb+edge_emb_b4), torch.relu(feature_emb+feature_emb_b4)
        
        d_hat = self.eph(node_emb, feature_emb)
        if len(self.cat_vars_pos) > 0: 
            d_hat[:, self.cat_vars_pos] = torch.sigmoid(d_hat[:, self.cat_vars_pos]) 

        d_hat_adj = d_hat.clone().detach()
        d_hat_adj[x['mask']==1] = x['x'][x['mask'] == 1]
        y_hat = self.nph(d_hat_adj)

        return {
            'x_imputed': d_hat,
            'x_impted_adj': d_hat_adj,
            'preds': y_hat
        }

    def train_step(self, batch): 
        # returns the training loss 
        # (1) feed forward
        # with torch.set_grad_enabled(True)
        out = self.forward(batch)
        mse_loss = nn.MSELoss()
        cls_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCELoss()

        if self.num_labels == 1: 
            label_loss = mse_loss(out['preds'], batch['y'])
        else: 
            label_loss = cls_loss(out['preds'], batch['y'])
        
        total_loss = label_loss
        
        if len(self.cat_vars_pos) > 0:
            cat_imp_loss = bce_loss(out['x_imputed'][:, self.cat_vars_pos], batch['x_complete'][:, self.cat_vars_pos])
            total_loss += self.imp_loss_penalty * cat_imp_loss
        else: 
            cat_imp_loss = float('nan')
        if len(self.numeric_vars_pos) > 0: 
            num_imp_loss = mse_loss(out['x_imputed'][:, self.numeric_vars_pos], batch['x_complete'][:, self.numeric_vars_pos])
            total_loss += self.imp_loss_penalty * num_imp_loss
        else: 
            num_imp_loss = float('nan')

        return {
            'x_imputed': out.get('x_imputed'),
            'x_impted_adj': out.get('x_imputed_adj'),
            'preds': out.get('preds'),
            'cat_imp_loss': cat_imp_loss,
            'num_imp_loss': num_imp_loss,
            'label_loss': label_loss,
            'total_loss': total_loss
        } 

    @torch.no_grad()
    def val_step(self, batch): 
        # with torch.no_grad()
        out = self.forward(batch)
        mse_loss = nn.MSELoss()
        cls_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCELoss()

        if self.num_labels == 1: 
            label_loss = mse_loss(out['preds'], batch['y'])
        else: 
            label_loss = cls_loss(out['preds'], batch['y'])
        
        total_loss = label_loss
        
        if len(self.cat_vars_pos) > 0:
            cat_imp_loss = bce_loss(out['x_imputed'][:, self.cat_vars_pos], batch['x_complete'][:, self.cat_vars_pos])
            total_loss += self.imp_loss_penalty * cat_imp_loss
        else: 
            cat_imp_loss = float('nan')
        if len(self.numeric_vars_pos) > 0: 
            num_imp_loss = mse_loss(out['x_imputed'][:, self.numeric_vars_pos], batch['x_complete'][:, self.numeric_vars_pos])
            total_loss += self.imp_loss_penalty * num_imp_loss
        else: 
            num_imp_loss = float('nan')

        return {
            'x_imputed': out.get('x_imputed'),
            'x_impted_adj': out.get('x_imputed_adj'),
            'preds': out.get('preds'),
            'cat_imp_loss': cat_imp_loss,
            'num_imp_loss': num_imp_loss,
            'label_loss': label_loss,
            'total_loss': total_loss
        } 

    @torch.no_grad()
    def test_step(self, batch): 
        # with torch.no_grad()
        # returns test loss
        # and test performance measures
        out = self.forward(batch)
        mse_loss = nn.MSELoss()
        cls_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCELoss()

        if self.num_labels == 1: 
            label_loss = mse_loss(out['preds'], batch['y'])
        else: 
            label_loss = cls_loss(out['preds'], batch['y'])
        
        total_loss = label_loss
        
        if len(self.cat_vars_pos) > 0:
            cat_imp_loss = bce_loss(out['x_imputed'][:, self.cat_vars_pos], batch['x_complete'][:, self.cat_vars_pos])
            total_loss += cat_imp_loss
            cat_imp_loss = cat_imp_loss.detach().cpu().numpy()
        else: 
            cat_imp_loss = float('nan')
        if len(self.numeric_vars_pos) > 0: 
            num_imp_loss = mse_loss(out['x_imputed'][:, self.numeric_vars_pos], batch['x_complete'][:, self.numeric_vars_pos])
            total_loss += num_imp_loss
            num_imp_loss = num_imp_loss.detach().cpu().numpy()
        else: 
            num_imp_loss = float('nan')

        total_loss = total_loss.detach().cpu().numpy()
        label_loss = label_loss.detach().cpu().numpy()
        y = batch['y'].detach().cpu().numpy()
        # if regression:
        # return r2-score, mae and mse 
        if self.num_labels == 1:
            preds = out['preds'].detach().cpu().numpy()
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds) 
            mse = mean_squared_error(y, preds)
            return {
            #     'x_imputed': out.get('x_imputed'),
            #     'x_impted_adj': out.get('x_imputed_adj'),
            #     'preds': out.get('preds'),
                'cat_imp_loss': cat_imp_loss,
                'num_imp_loss': num_imp_loss,
                'label_loss': label_loss,
                'total_loss': total_loss,
                'r2':r2,
                'mae':mae,
                'mse':mse
            } 
        # elif classification 
        # return accuracy, precision, f1-score, 
        else: 
            preds = torch.argmax(F.softmax(out['preds'], dim=1), dim=1).detach().cpu().numpy()
            labels = np.array([np.arange(self.num_labels)]) 
            cm = confusion_matrix(y, preds, labels= labels)
            acc, rec, prec, f1 = evaluate(cm, weighted= False) 
            return {
                # 'x_imputed': out.get('x_imputed'),
                # 'x_impted_adj': out.get('x_imputed_adj'),
                # 'preds': out.get('preds'),
                'cat_imp_loss': cat_imp_loss,
                'num_imp_loss': num_imp_loss,
                'label_loss': label_loss,
                'total_loss': total_loss,
                'accuracy': acc,
                'recall': rec, 
                'precision': prec,
                'f1_score': f1
            } 


