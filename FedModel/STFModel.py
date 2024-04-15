import torch
from FedModel.Attention_Model import Attention, S_Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HGTConv

class SpatialTemporalFusionModel_Warehouse(torch.nn.Module):
    def __init__(self,
                hidden_channels=32,
                 ):
        super(SpatialTemporalFusionModel_Warehouse,self).__init__()
        self.hidden_size = hidden_channels
        self.att_w_up = S_Attention(ori_size=hidden_channels, q_size= hidden_channels,k_size=2*hidden_channels,v_size=2*hidden_channels,hid=hidden_channels)
        

    def forward(self,xs,xt):
        xup_w = self.att_w_up(xs,xs,xt,xt)
        xs = xup_w
        return xs

class SpatialTemporalFusionModel_SortingCenter(torch.nn.Module):
    def __init__(self,
                hidden_channels=32,
                 ):
        super(SpatialTemporalFusionModel_SortingCenter,self).__init__()
        self.hidden_size = hidden_channels
        self.att_s_up = S_Attention(ori_size=hidden_channels, q_size= 2*hidden_channels,k_size= hidden_channels,v_size=hidden_channels,hid=hidden_channels)

    def forward(self,xs,xt):
        xup_s = self.att_s_up(xs,xt,xs,xs)
        xs = xup_s
        return xs