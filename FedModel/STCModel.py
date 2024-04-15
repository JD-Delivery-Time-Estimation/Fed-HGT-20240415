import torch
from FedModel.Attention_Model import Attention, S_Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HGTConv


class SpatialTemporalCorrelationModel_Warehouse(torch.nn.Module):
    def __init__(self,
                hidden_channels=32,
                ) :
        super(SpatialTemporalCorrelationModel_Warehouse,self).__init__()
        self.hidden_size = hidden_channels
        self.att_w = Attention(q_size=2 *hidden_channels,k_size=hidden_channels,v_size=hidden_channels,hid=hidden_channels)
       
    def forward(self,xs,xt):
        xst_w = self.att_w(xt,xs,xs)
        
        xst = xst_w
        
        return xst

class SpatialTemporalCorrelationModel_SortingCenter(torch.nn.Module):
    def __init__(self,
                hidden_channels=32,
                ) :
        super(SpatialTemporalCorrelationModel_SortingCenter,self).__init__()
        self.hidden_size = hidden_channels
        self.att_s = Attention(q_size= hidden_channels,k_size= 2*hidden_channels,v_size=2*hidden_channels,hid=hidden_channels)

    def forward(self,xs,xt):
        xst_s = self.att_s(xs,xt,xt)
        xst = xst_s
        
        return xst

