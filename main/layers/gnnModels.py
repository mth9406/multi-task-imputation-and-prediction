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
                num_layers, 
                node_emb_size,
                edge_emb_size,
                msg_emb_size,
                num_features,
                num_labels,
                edge_drop_p,
                device
                ):
        super().__init__() 
        
        self.init = Init(num_features, device= device)
        for i in num_layers: 
            setattr(self, f'gcn_block{i}', GCNBlock(node_emb_size, edge_emb_size, msg_emb_size))
        self.eph = EdgePredictionHead(node_emb_size, num_features, device)
        self.nph = NodePredictionHead(num_features, num_labels)

        self.num_layers = num_layers
        self.edge_drop_p = edge_drop_p
        self.node_emb_size = node_emb_size
        self.edge_emb_size = edge_emb_size
        self.msg_emb_size = msg_emb_size
        self.num_features = num_features
        self.num_labels = num_labels 
        self.device = device

    def forward(self, x, edge_index):
        
        return 

    def train_step(self, batch): 
        # returns the training loss 
        # (1) feed forward
        # with torch.set_grad_enabled(True)
        return {
            'imp_loss': None,
            'label_loss': None,
            'total_loss': None
        } 

    @torch.no_grad()
    def val_step(self, batch): 
        # with torch.no_grad()
        # returns validation loss
        # and validation performance measures
        return {
            'imp_loss': None,
            'label_loss': None,
            'total_loss': None
        } 

    @torch.no_grad()
    def test_step(self, batch): 
        # with torch.no_grad()
        # returns test loss
        # and test performance measures
        return {
            'imp_loss': None,
            'label_loss': None,
            'total_loss': None
        } 


