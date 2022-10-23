import torch
from torch import nn 
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
# from torch_geometric.nn import MessagePassing

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_sum
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GATConv 

from fancyimpute import SoftImpute

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

class InitLNF():
    r"""Initialize node features 
    returns node_features and edge_features 
    """
    def __init__(self, num_features, device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__() 
        self.num_features = num_features 
        self.onehot_vector = torch.eye(num_features, requires_grad= False).to(device)
        self.device = device 

    def __call__(self, batch_size, x): 
        node_feature = torch.FloatTensor(SoftImpute(verbose= False).fit_transform(x.detach().cpu())).to(self.device)
        return node_feature, self.onehot_vector

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
        feature_emb: feature embedding (num_features, node_emb_size)
        edge_index: edge index (adj matrix in COO format) (2, num_edges)
        """
        src, dst = edge_index
        # generate messages for observation update and feature update
        msg = torch.concat([feature_emb[dst], edge_emb], dim= 1) # num_edges, node_emb_size+edge_emb_size 
        msg = torch.relu(msg@self.P + self.b_P) # num_edge, msg_emb_size
        msg = scatter_mean(msg, src, dim= 0) # bs, msg_emb_size  

        msg_f = torch.concat([node_emb[src], edge_emb], dim= 1) # num_edges, node_emb_size+edge_emb_size 
        msg_f = torch.relu(msg_f@self.P + self.b_P) # num_edge, msg_emb_size
        msg_f = scatter_mean(msg_f, dst, dim= 0) # num_features, msg_emb_size 
        # the number of nodes = bs + num_features

        # node update 
        node_emb = torch.concat([node_emb, msg], dim=1)
        node_emb = node_emb@self.Q + self.b_Q # bs, node_emb_size 
        feature_emb = torch.concat([feature_emb, msg_f], dim=1)
        feature_emb = feature_emb@self.Q + self.b_Q # num_features, node_emb_size 
           
        # edge update
        edge_emb = torch.concat([edge_emb, node_emb[src], feature_emb[dst]], dim= 1)
        edge_emb = edge_emb@self.W + self.b_W

        return node_emb, edge_emb, feature_emb

class GCNBlockVer2(nn.Module):
    def __init__(self, in_node_emb_size, out_node_emb_size, in_edge_emb_size, out_edge_emb_size, msg_emb_size):
        super().__init__()
        
        # define trainable parameters
        self.P_src = nn.Parameter(torch.randn(size= (in_node_emb_size+in_edge_emb_size, msg_emb_size), requires_grad= True)) # for msg
        self.b_P_src = nn.Parameter(torch.zeros(size= (msg_emb_size, ), requires_grad= True))
        self.P_dst = nn.Parameter(torch.randn(size= (in_node_emb_size+in_edge_emb_size, msg_emb_size), requires_grad= True)) 
        self.b_P_dst = nn.Parameter(torch.zeros(size= (msg_emb_size, ), requires_grad= True))

        self.Q_src_self = nn.Parameter(torch.randn(size= (in_node_emb_size, out_node_emb_size), requires_grad= True)) # for node update
        self.b_Q_src_self = nn.Parameter(torch.zeros(size= (out_node_emb_size,), requires_grad= True))
        self.Q_src_neigh = nn.Parameter(torch.randn(size= (msg_emb_size, out_node_emb_size), requires_grad= True))
        self.b_Q_src_neigh = nn.Parameter(torch.zeros(size= (out_node_emb_size,), requires_grad= True))
        
        self.Q_dst_self = nn.Parameter(torch.randn(size= (in_node_emb_size, out_node_emb_size), requires_grad= True)) 
        self.b_Q_dst_self = nn.Parameter(torch.zeros(size= (out_node_emb_size,), requires_grad= True))
        self.Q_dst_neigh = nn.Parameter(torch.randn(size= (msg_emb_size, out_node_emb_size), requires_grad= True))
        self.b_Q_dst_neigh = nn.Parameter(torch.zeros(size= (out_node_emb_size,), requires_grad= True))

        self.W = nn.Parameter(torch.randn(size= (in_edge_emb_size+out_node_emb_size+out_node_emb_size, out_edge_emb_size), requires_grad= True)) # for edge update
        self.b_W = nn.Parameter(torch.zeros(size= (out_edge_emb_size, ), requires_grad= True))

        self.init_params()
        
        self.in_node_emb_size = in_node_emb_size
        self.out_node_emb_size = out_node_emb_size
        self.in_edge_emb_size = in_edge_emb_size
        self.out_edge_emb_size = out_edge_emb_size
        self.msg_emb_size = msg_emb_size

    def init_params(self): 
        nn.init.xavier_normal_(self.P_src)
        nn.init.xavier_normal_(self.P_dst)
        nn.init.xavier_normal_(self.Q_src_self)
        nn.init.xavier_normal_(self.Q_src_neigh)
        nn.init.xavier_normal_(self.Q_dst_self)
        nn.init.xavier_normal_(self.Q_dst_neigh)
        nn.init.xavier_normal_(self.W)


    def forward(self, node_emb, edge_emb, feature_emb, edge_index):
        r"""
        node_emb: node embedding (bs, node_emb_size)
        edge_emb: edge embedding (num_edges, edge_emb_size)
        feature_emb: feature embedding (num_features, node_emb_size)
        edge_index: edge index (adj matrix in COO format) (2, num_edges)
        """
        src, dst = edge_index
        # generate messages for observation update and feature update
        msg = torch.concat([feature_emb[dst], edge_emb], dim= 1) # num_edges, node_emb_size+edge_emb_size 
        msg = F.leaky_relu(msg@self.P_src + self.b_P_src) # num_edge, msg_emb_size
        msg = scatter_mean(msg, src, dim= 0) # bs, msg_emb_size  

        msg_f = torch.concat([node_emb[src], edge_emb], dim= 1) # num_edges, node_emb_size+edge_emb_size 
        msg_f = F.leaky_relu(msg_f@self.P_dst + self.b_P_dst) # num_edge, msg_emb_size
        msg_f = scatter_mean(msg_f, dst, dim= 0) # num_features, msg_emb_size 
        # the number of nodes = bs + num_features

        # node update 
        node_emb =  node_emb@self.Q_src_self + self.b_Q_src_self + msg@self.Q_src_neigh + self.b_Q_src_neigh # bs, node_emb_size 
        feature_emb =  feature_emb@self.Q_dst_self + self.b_Q_dst_self + msg_f@self.Q_dst_neigh + self.b_Q_dst_neigh # num_features, node_emb_size 
           
        # edge update
        edge_emb = torch.concat([edge_emb, node_emb[src], feature_emb[dst]], dim= 1)
        edge_emb = edge_emb@self.W + self.b_W

        return node_emb, edge_emb, feature_emb

class EdgePredictionHead(nn.Module): 

    def __init__(self, node_emb_size, num_features, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__() 
        if node_emb_size//4 > 1: 
            self.decode = nn.Sequential(
                nn.Linear(2*node_emb_size, node_emb_size//2),
                nn.BatchNorm1d(node_emb_size//2), 
                nn.LeakyReLU(),
                nn.Linear(node_emb_size//2, node_emb_size//4),
                nn.BatchNorm1d(node_emb_size//4),
                nn.LeakyReLU(),
                nn.Linear(node_emb_size//4, 1)
                )
        elif node_emb_size//2 > 1: 
            self.decode = nn.Sequential(
                nn.Linear(2*node_emb_size, node_emb_size//2),
                nn.BatchNorm1d(node_emb_size//2), 
                nn.LeakyReLU(),
                nn.Linear(node_emb_size//2, 1)
                )         
        else: 
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

def coo_to_adj(rel, num_objects, device= None):
    r"""Converts COO-type to adjacency matrix
    
    # Arguments          
    ___________ 
    rel : torch.FloatTensor 
        relation - tensor                  
    num_objects : int
        the number of objects               
    device : torch.device     
        device - cpu or cuda    

    # Returns    
    _________     
    adj : torch.FloatTensor
        adjacency matrix  
        a square matrix of shape: num_object x num_object 
    """

    adj = torch.full((num_objects*num_objects,), -float("inf")+1.)
    for i in range(num_objects-1):
        adj[i*num_objects+i+1:(i+1)*num_objects+(i+1)] = rel[i*num_objects:(i+1)*num_objects]
    if device is not None:
        adj = adj.to(device)
    return adj.reshape((num_objects,num_objects))

def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return torch.softmax(y / tau, dim= 1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def kl_categorical_uniform(preds, num_features, eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    # if add_const:
    #     const = np.log(num_edge_types)
    #     kl_div += const
    return -kl_div.sum() / (num_features * preds.size(0))  

def encode_onehot(labels): 
    r""" Encode some relational masks specifying which vertices receive messages from which other ones.
    # Arguments          
    ___________             
    labels : np.array type 
    
    # Returns        
    _________          
    labels_one_hot : np.array type            
        adjacency matrix
    
    # Example-usage       
    _______________            
    >>> labels = [0,0,0,1,1,1,2,2,2]
    >>> labels_onehot = encode_onehot(labels)
    >>> labels_onehot 
    array(
        [[1, 0, 0],
         [1, 0, 0],             
         [1, 0, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 0, 1],
         [0, 0, 1],            
         [0, 0, 1]], dtype=int32)      
    """
    classes = set(labels) 
    classes_dict = {c: np.identity(len(classes))[i,:] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype= np.int32)

def generate_off_diag(num_objects, device= None): 
    r"""Generates off-diagonal graph 
    
    # Arguments          
    ___________                
    num_objects : int
        the number of objects               
    device : torch.device     
        device - cpu or cuda    

    # Returns    
    _________     
    rel_rec : torch.FloatTensor    
        relation-receiver     
    rel_send : torch.FloatTensor    
        relation-receiver     
    """
    off_diag = np.ones([num_objects, num_objects]) - np.eye(num_objects)

    rec, send = np.where(off_diag)
    rel_rec = np.array(encode_onehot(rec), dtype= np.float32)
    rel_send = np.array(encode_onehot(send), dtype= np.float32)
    
    rel_rec = torch.FloatTensor(rel_rec) if device is None \
        else torch.FloatTensor(rel_rec).to(device)
    rel_send = torch.FloatTensor(rel_send) if device is None \
        else torch.FloatTensor(rel_send).to(device)
    
    return rel_rec, rel_send

class GraphLearningLayer(nn.Module): 

    def __init__(self, num_features, 
                in_feature_emb_size,            
                out_feature_emb_size,
                prior_relation_index,
                tau= 0.1, device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__() 
        self.num_features = num_features
        self.tau = tau
        self.device = device
        self.prior_relation_index = prior_relation_index
        self.prior_adj = to_dense_adj(prior_relation_index)

        self.edge = self.get_edges_index()
        # self.edge_id = torch.arange(num_features*(num_features-1)).to(device)
        self.proj = nn.Linear(in_feature_emb_size, out_feature_emb_size)
        self.node2edge = nn.Linear(2*out_feature_emb_size, out_feature_emb_size)
        self.edge2node = nn.Linear(out_feature_emb_size, out_feature_emb_size)
        self.node2edge2 = nn.Linear(2*out_feature_emb_size, out_feature_emb_size)
        self.fc_out = nn.Linear(2*out_feature_emb_size, 1)

    def forward(self, feature_emb): 
        h = self.proj(feature_emb) # num_features, out_feature_emb_size
        src, dst = self.edge
        # (1) node to edge
        h_e = F.leaky_relu(self.node2edge(torch.concat([h[src], h[dst]], axis=1))) # num_edges, out_feature_emb_size
        h_e_skip = h_e
        # (2) edge to node
        h = F.leaky_relu(self.edge2node(scatter_mean(h_e, src, dim=0))) # num_features, out_feature_emb_size
        # (3) node to edge
        h_e = F.leaky_relu(self.node2edge2(torch.concat([h[src], h[dst]], axis=1))) # num_edges, out_feature_emb_size
        
        # (4) 
        h_e = self.fc_out(torch.concat([h_e, h_e_skip], dim=1)) # num_edges, 1
        logits = coo_to_adj(h_e.squeeze(), self.num_features, self.device)
        
        if self.training:
            relation = gumbel_softmax(logits, self.tau, hard= True)
            # relation = relation.fill_diagonal_(0.)
            probs = torch.softmax(logits, dim=1)
            kl_loss = kl_categorical_uniform(probs, self.num_features)
        else: 
            relation = gumbel_softmax(logits, self.tau, hard= True)
            # relation = relation.fill_diagonal_(0.)
            kl_loss = None
        relation_index = torch.nonzero(relation).T
        relation_index = relation_index.to(relation.device)
        return {
            'relation': relation,
            'relation_index': relation_index,
            'kl_loss': kl_loss
        }

    def get_edges_index(self): 
        adj = torch.ones((self.num_features, self.num_features)) - torch.eye(self.num_features)
        edges = (torch.nonzero(adj).T).to(self.device)
        return edges

def gumbel_sigmoid_sample(logits, tau=0.1, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return torch.sigmoid(y / tau)

def gumbel_sigmoid(logits, tau=1, hard=False, eps=1e-10):
    y_soft = gumbel_sigmoid_sample(logits, tau=tau, eps=eps)
    if hard:
        return (y_soft > 0.5) * 1.
    else:
        y = y_soft
    return y

def prior_fitting_loss(probs, prior_adj, eps=1e-10): 
    kl_div = -prior_adj * torch.log(probs + eps)
    return kl_div.mean()

class AdaptiveGraphLearningLayer(nn.Module): 

    def __init__(self, num_features, prior_relation_index, tau= 0.1):
        super().__init__() 
        self.num_features = num_features
        self.tau = tau
        self.prior_relation_index = prior_relation_index
        self.prior_adj = to_dense_adj(prior_relation_index)

    def forward(self, feature_emb): 
        # returns relationship between feature embeddings 
        # transform the output by torch.nonzero into edge_index 
        # relationship is computed as 
        # 1) correlation 
        # or 
        # 2) sth else from the GRL book. 
        # feature_emb: (num_features, node_emb_size)
        logits = feature_emb@feature_emb.T # num_features, num_features
        if self.training:
            relation = gumbel_sigmoid(logits, self.tau, hard= False)
            # relation = relation.fill_diagonal_(0.)
            kl_loss = prior_fitting_loss(relation, self.prior_adj)
        else: 
            relation = gumbel_softmax(logits, self.tau, hard= True)
            # relation = relation.fill_diagonal_(0.)
            kl_loss = None
        relation_index = torch.nonzero(relation).T
        relation_index = relation_index.to(relation.device)
        return {
            'relation': relation,
            'relation_index': relation_index,
            'kl_loss': kl_loss
        }

class AttentionEdgePredictionHead(nn.Module): 

    def __init__(self, node_emb_size, num_features, heads:int= 2, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__() 
        
        self.gat = GATConv(node_emb_size, node_emb_size//2, heads= heads, add_self_loops= False)
        msg_emb_size = (node_emb_size//2) * heads
        in_channels = msg_emb_size + node_emb_size
        if in_channels//4 > 1: 
            self.decode = nn.Sequential(
                nn.Linear(in_channels, in_channels//2),
                nn.BatchNorm1d(in_channels//2), 
                nn.LeakyReLU(),
                nn.Linear(in_channels//2, in_channels//4), 
                nn.BatchNorm1d(in_channels//4),
                nn.LeakyReLU(),
                nn.Linear(in_channels//4, 1)
                )
        elif in_channels//2 > 1: 
            self.decode = nn.Sequential(
                nn.Linear(in_channels, in_channels//2),
                nn.BatchNorm1d(in_channels//2), 
                nn.LeakyReLU(),
                nn.Linear(in_channels//2, 1)
                )         
        else: 
            self.decode = nn.Linear(in_channels, 1)

        self.node_emb_size = node_emb_size
        self.msg_emb_size = msg_emb_size
        self.num_features = num_features
        self.in_channels = in_channels
        self.device = device

    def forward(self, node_emb, feature_emb, relation_index):
        
        bs = node_emb.shape[0]
        src, dst = self.get_edges_index(bs)
        
        msg = self.gat(feature_emb, relation_index)
        msg = torch.concat([node_emb[src], msg[dst]], dim= -1) # num_edges, msg_emb_size+node_emb_size
        msg = self.decode(msg)

        return msg.reshape(-1, self.num_features)

    def get_edges_index(self, bs): 
        nodes = torch.arange(bs) 
        features = torch.arange(self.num_features) 
        src = nodes.repeat_interleave(self.num_features).to(self.device)
        dst = features.repeat(bs).to(self.device)

        return src, dst

class RelationalEdgePredictionHead(nn.Module): 

    def __init__(self, node_emb_size, num_features, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__() 
        
        if node_emb_size//4 > 1: 
            self.decode = nn.Sequential(
                nn.Linear(2*node_emb_size, node_emb_size//2),
                nn.BatchNorm1d(node_emb_size//2), 
                nn.Linear(node_emb_size//2, node_emb_size//4), 
                nn.BatchNorm1d(node_emb_size//4),
                nn.Linear(node_emb_size//4, 1)
                )
        elif node_emb_size//2 > 1: 
            self.decode = nn.Sequential(
                nn.Linear(2*node_emb_size, node_emb_size//2),
                nn.BatchNorm1d(node_emb_size//2), 
                nn.Linear(node_emb_size//2, 1)
                )         
        else: 
            self.decode = nn.Linear(2*node_emb_size, 1)
        self.node_emb_size = node_emb_size
        self.num_features = num_features
        self.device = device

    def forward(self, node_emb, feature_emb, relation_index):
        bs = node_emb.shape[0]
        src, dst, src_neigh, dst_neigh, agg_by = self.get_neighbors(relation_index, bs)  

        msg = scatter_mean(feature_emb[dst_neigh], agg_by, dim= 0) # num_edges (=bs*num_features), feature_emb_size
        msg = torch.concat([node_emb[src], msg], dim= -1)     
        msg = self.decode(msg)

        return msg.reshape(-1, self.num_features)

    def get_neighbors(self, relation_index, bs): 
        nodes = torch.arange(bs) 
        features = torch.arange(self.num_features)        
        src = nodes.repeat_interleave(self.num_features).to(self.device)
        dst = features.repeat(bs).to(self.device)
        src_r, dst_r = relation_index 
        agg_by = []
        src_neigh, dst_neigh = [], []
        for i in range(len(src)): 
            src_neigh.append(src[i].detach().cpu().item())
            dst_neigh.append(dst[i].detach().cpu().item())
            related_features = dst_r[src_r==dst[i]].tolist()
            srcs = [src[i].detach().cpu().item() for _ in range(len(related_features))]
            agg = [i for _ in range(len(related_features)+1)]
            dst_neigh += related_features
            src_neigh += srcs
            agg_by += agg
        
        src_neigh, dst_neigh, agg_by =\
            torch.tensor(src_neigh).to(self.device),torch.tensor(dst_neigh).to(self.device),torch.tensor(agg_by).to(self.device)
        
        return src, dst, src_neigh, dst_neigh, agg_by