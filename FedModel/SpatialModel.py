import torch
from FedModel.Attention_Model import Attention, S_Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HGTConv

class HeterogeneousSpatialGNN(torch.nn.Module):
    def __init__(self,
                data,
                hidden_channels=32, 
                num_heads=2, 
                num_layers=2
                ):
        
        super(HeterogeneousSpatialGNN,self).__init__()

        self.hidden_size = hidden_channels
        self.lin_dict = torch.nn.ModuleDict()

        # Extract Spatial Feature
        for node_type in data.node_types:
            self.lin_dict[node_type] =  Linear(-1, hidden_channels)
        
        # Graph Transformer
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)
        
    def forward(self,x_dict,edge_index_dict):
        '''
        Given u^{Spatial} at time t
        Output u^{Spatial \prime} at time t
        '''


        out = {}

        for node_type, x in x_dict.items():
            out[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            out = conv(out, edge_index_dict)
        
        return out
        

    



