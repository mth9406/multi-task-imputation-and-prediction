import torch
from torch import nn 
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_sum

class Init(object): 
    r"""Initialize node features 
    returns node_features and edge_features  
    """
    def __init__(self, num_features, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')): 
        super().__init__() 
        self.num_features = num_features
        self.const_vector = torch.ones(1, num_features, requires_grad= False).to(device) 
        self.onehot_vector = torch.eye(num_features, requires_grad= False).to(device) # num_feature, num_feature 
        self.device = device

    def __call__(self, batch_size): 
        return self.const_vector.repeat((batch_size,1)), self.onehot_vector

class GCNBlock(nn.Module):
    def __init__(self, in_node_emb_size, out_node_emb_size, in_edge_emb_size, out_edge_emb_size, msg_emb_size):
        super().__init__()
        
        # define trainable parameters
        self.P = nn.Parameter(torch.randn(size= (in_node_emb_size+in_edge_emb_size, msg_emb_size), requires_grad= True)) # for msg
        self.Q = nn.Parameter(torch.randn(size= (in_node_emb_size+msg_emb_size, out_node_emb_size), requires_grad= True)) # for node update
        self.W = nn.Parameter(torch.randn(size= (in_edge_emb_size+out_node_emb_size+out_node_emb_size, out_edge_emb_size), requires_grad= True)) # for edge update
        self.init_params()
        self.b_P = nn.Parameter(torch.zeros(size= (msg_emb_size, ), requires_grad= True))
        self.b_Q = nn.Parameter(torch.zeros(size= (out_node_emb_size, ), requires_grad= True))
        self.b_W = nn.Parameter(torch.zeros(size= (out_edge_emb_size, ), requires_grad= True))

    def init_params(self): 
        nn.init.xavier_normal_(self.P)
        nn.init.xavier_normal_(self.Q)
        nn.init.xavier_normal_(self.W)

    def forward(self, node_emb, edge_emb, feature_emb, edge_index):
        r"""
        node_emb: node embedding (bs, node_emb_size)
        edge_emb: edge embedding (num_edges, edge_emb_size)
        edge_index: edge index (adj matrix in COO format) (2, num_edges)
        """
        src, dst = edge_index
        # generate messages for observation update and feature update
        msg = torch.concat([feature_emb[dst], edge_emb], dim= 1) # num_edges, node_emb_size+edge_emb_size 
        msg = torch.relu(msg@self.P + self.b_P) # num_edge, msg_emb_size
        msg = scatter_mean(msg, src, dim= 0) # bs, msg_emb_size  

        msg_f = torch.concat([node_emb[src], edge_emb], dim= 1) # num_edges, node_emb_size+edge_emb_size 
        mgs_f = torch.relu(msg_f@self.P + self.b_P) # num_edge, msg_emb_size
        mgs_f = scatter_mean(mgs_f, dst, dim= 0) # num_features, msg_emb_size 
        # the number of nodes = bs + num_features

        # node update 
        node_emb = torch.concat([node_emb, msg], dim=1)
        node_emb = node_emb@self.Q + self.b_Q # bs, node_emb_size 
        feature_emb = torch.concat([feature_emb, mgs_f], dim=1)
        feature_emb = feature_emb@self.Q + self.b_Q # num_features, node_emb_size 

        # edge update
        edge_emb = torch.concat([edge_emb, node_emb[src], feature_emb[dst]], dim= 1)
        edge_emb = edge_emb@self.W + self.b_W

        return node_emb, edge_emb, feature_emb

class EdgePredictionHead(nn.Module): 

    def __init__(self, node_emb_size, num_features, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__() 
        self.decode = nn.Linear(2*node_emb_size, 1)
        self.node_emb_size = node_emb_size
        self.num_features = num_features
        self.device = device

    def forward(self, node_emb, feature_emb):
        bs = node_emb.shape[0]
        nodes = torch.arange(bs) 
        features = torch.arange(self.num_features)
        src = nodes.repeat_interleave(self.num_features).to(self.device)
        dst = features.repeat(bs).to(self.device)
        return self.decode(torch.concat([node_emb[src], feature_emb[dst]], dim= 1)).reshape(-1, self.num_features)

class NodePredictionHead(nn.Module): 

    def __init__(self, num_features, num_labels): 
        super().__init__() 
        self.decode = nn.Linear(num_features, num_labels)
        self.num_features = num_features 
        self.num_labels = num_labels
    
    def forward(self, x_imputed): 
        if self.num_labels == 1:
            return self.decode(x_imputed).flatten() 
        else: 
            return self.decode(x_imputed)