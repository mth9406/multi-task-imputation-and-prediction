from mimetypes import init
import torch 
from torch import nn
from torch.nn import functional as F
from layers.gnnLayers import *
from torch_geometric.nn import (
    SAGEConv,
    Aggregation,
    MeanAggregation,
    MaxAggregation,
    SumAggregation,
    StdAggregation,
    VarAggregation,
    MultiAggregation,
    SoftmaxAggregation,
)
from torch_geometric.utils import dropout_adj

# Sample model 
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
        self.device = device

    def forward(self, x):
        edge_index = x['edge_index']
        edge_emb = x['edge_value'][:, None]
        node_emb, feature_emb = self.init(x['x'].shape[0])

        for i in range(self.num_layers):
            node_emb_b4, edge_emb_b4, feature_emb_b4 = node_emb.clone().detach(), edge_emb.clone().detach(), feature_emb.clone().detach()
            node_emb, edge_emb, feature_emb = getattr(self, f'gcn_block{i}')(node_emb, edge_emb, feature_emb, edge_index)
            node_emb, edge_emb, feature_emb \
                = torch.relu(node_emb+node_emb_b4), torch.relu(edge_emb+edge_emb_b4), torch.relu(feature_emb+feature_emb_b4)
        
        d_hat = self.eph(node_emb, feature_emb)
        y_hat = self.nph(d_hat)

        return {
            'x_imputed': d_hat,
            'preds': y_hat
        }

    def train_step(self, batch): 
        # returns the training loss 
        # (1) feed forward
        # with torch.set_grad_enabled(True)
        out = self.forward(batch)
        mse_loss = nn.MSELoss()
        cls_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCEWithLogitsLoss()

        if self.num_labels == 1: 
            label_loss = mse_loss(out['preds'], batch['y'])
        else: 
            label_loss = cls_loss(out['preds'], batch['y'])
        
        total_loss = label_loss
        
        if len(self.cat_vars_pos) > 0:
            num_imp_loss = mse_loss(out['x_imputed'][:, self.cat_vars_pos], batch['x_complete'][:, self.cat_vars_pos])
            total_loss += num_imp_loss
        else: 
            num_imp_loss = float('nan')
        if len(self.numeric_vars_pos) > 0: 
            cat_imp_loss = bce_loss(out['x_imputed'][:, self.cat_vars_pos], batch['x_complete'][:, self.cat_vars_pos])
            total_loss += cat_imp_loss
        else: 
            cat_imp_loss = float('nan')

        return {
            'x_imputed': out.get('x_imputed'),
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
        bce_loss = nn.BCEWithLogitsLoss()

        if self.num_labels == 1: 
            label_loss = mse_loss(out['preds'], batch['y'])
        else: 
            label_loss = cls_loss(out['preds'], batch['y'])
        
        total_loss = label_loss
        
        if len(self.cat_vars_pos) > 0:
            num_imp_loss = mse_loss(out['x_imputed'][:, self.cat_vars_pos], batch['x_complete'][:, self.cat_vars_pos])
            total_loss += num_imp_loss
        else: 
            num_imp_loss = float('nan')
        if len(self.numeric_vars_pos) > 0: 
            cat_imp_loss = bce_loss(out['x_imputed'][:, self.cat_vars_pos], batch['x_complete'][:, self.cat_vars_pos])
            total_loss += cat_imp_loss
        else: 
            cat_imp_loss = float('nan')

        return {
            'x_imputed': out.get('x_imputed'),
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
        bce_loss = nn.BCEWithLogitsLoss()

        if self.num_labels == 1: 
            label_loss = mse_loss(out['preds'], batch['y'])
        else: 
            label_loss = cls_loss(out['preds'], batch['y'])
        
        total_loss = label_loss
        
        if len(self.cat_vars_pos) > 0:
            num_imp_loss = mse_loss(out['x_imputed'][:, self.cat_vars_pos], batch['x_complete'][:, self.cat_vars_pos])
            total_loss += num_imp_loss
        else: 
            num_imp_loss = float('nan')
        if len(self.numeric_vars_pos) > 0: 
            cat_imp_loss = bce_loss(out['x_imputed'][:, self.cat_vars_pos], batch['x_complete'][:, self.cat_vars_pos])
            total_loss += cat_imp_loss
        else: 
            cat_imp_loss = float('nan')

        # if regression:
        # return r2-score, mae and mse 

        # elif classification 
        # return accuracy, precision, f1-score, 

        return {
            'x_imputed': out.get('x_imputed'),
            'preds': out.get('preds'),
            'cat_imp_loss': cat_imp_loss,
            'num_imp_loss': num_imp_loss,
            'label_loss': label_loss,
            'total_loss': total_loss
        } 


