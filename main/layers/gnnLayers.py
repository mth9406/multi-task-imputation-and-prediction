import torch
from torch import nn 
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing

# import torch_scatter 

# class ExampleLayer(MessagePassing):
    
#       def __init__(self, in_channels, out_channels, normalize = True,
#                   bias = False, **kwargs):  
#             super(ExampleLayer, self).__init__(**kwargs)

#             self.in_channels = in_channels
#             self.out_channels = out_channels
#             self.normalize = normalize

#             self.lin_l = nn.Linear(self.in_channels, self.out_channels) 
#             # linear transformation that you apply to embedding  for central node.
                
#             self.lin_r = nn.Linear(self.in_channels, self.out_channels) 
#             # linear transformation that you apply to aggregated(already) info from neighbors.

#             self.reset_parameters()

#       def reset_parameters(self):
#           self.lin_l.reset_parameters()
#           self.lin_r.reset_parameters()      

#       def forward(self, x, edge_index, size = None):
#             prop = self.propagate(edge_index, x=(x, x), size=size) 
#             # see Messsage.Passing.propagate() in https://pytorch-geometric.readthedocs.io/en/latest/
#             out = self.lin_l(x) + self.lin_r(prop)
#             if self.normalize:
#               out = F.normalize(out, p=2)
            
#             return out
      
#       # Implement your message function here.
#       def message(self, x_j):
#           out = x_j
#           return out
      
#       # Implement your aggregate function here.
#       def aggregate(self, inputs, index, dim_size = None):
#             # The axis along which to index number of nodes.
#             node_dim = self.node_dim
#             # since 
#             out = torch_scatter.scatter(inputs, index, node_dim, dim_size=dim_size, reduce='mean') 
#             # see https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html#torch_scatter.scatter
#             return out

class GCNBlock(MessagePassing): 

    def __init__(self, num_layers, in_channels, out_channels, **kwargs): 
          
          super().__init__() 




      
