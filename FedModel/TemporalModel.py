import torch
from FedModel.Attention_Model import Attention, S_Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HGTConv

class IntialedGRUwithBackgroundInformation(torch.nn.Module):
    def __init__(self,
                hidden_channels=32,
                temporal_features = 4,
                back_features = 3
                ):
        super(IntialedGRUwithBackgroundInformation,self).__init__()
        self.hidden_size = hidden_channels
        self.gru_in_fc= Linear(temporal_features,hidden_channels)
        self.gru_init_fc = Linear(hidden_channels,hidden_channels)
        self.gru = nn.GRU(input_size=hidden_channels,hidden_size=hidden_channels, batch_first=True)
        self.back_fc = Linear(back_features,hidden_channels)
        
        self.skip_gru_hidden = None
        
        
        self.skip_gru_transformation = Linear(4*hidden_channels,2*hidden_channels)

    def forward(self,
               train_temporal_data,
               train_back_data,
               t,
               skip_gru_hidden,
               if_first=False,
               init_spatial_features=None):
        self.skip_gru_hidden = skip_gru_hidden
        gru_in = self.gru_in_fc(train_temporal_data).view(-1,1,self.hidden_size)
        gru_out1 = None
        gru_out2 = None
        gru_out3 = None
        


        if if_first and init_spatial_features is not None:
            gru_init = self.gru_init_fc(init_spatial_features).view(1,-1,self.hidden_size)
            gru_out1, gru_hid= self.gru(gru_in,gru_init)
            self.skip_gru_hidden[t] = gru_hid
            if t < 24:
                gru_out2 = gru_out1
                gru_out3 = gru_out1
            if t >= 24 and t < 7 *24:
                gru_out3 = gru_out1
                gru_out2,_ = self.gru(gru_in,self.skip_gru_hidden[t-24].view(1,-1,self.hidden_size))
            if t >= 7 * 24:
                gru_out2,_ = self.gru(gru_in,self.skip_gru_hidden[t-24].view(1,-1,self.hidden_size))
                gru_out3,_ = self.gru(gru_in,self.skip_gru_hidden[t-7*24].view(1,-1,self.hidden_size))

        else:
            gru_out1,gru_hid = self.gru(gru_in)
            self.skip_gru_hidden[t] = gru_hid
            if t < 24:
                gru_out2 = gru_out1
                gru_out3 = gru_out1
            if t >= 24 and t < 7 *24:
                gru_out3 = gru_out1
                gru_out2, _ = self.gru(gru_in,self.skip_gru_hidden[t-24].view(1,-1,self.hidden_size))
            if t >= 7 * 24:
                gru_out2, _ = self.gru(gru_in,self.skip_gru_hidden[t-24].view(1,-1,self.hidden_size))
                gru_out3, _ = self.gru(gru_in,self.skip_gru_hidden[t-7*24].view(1,-1,self.hidden_size))
        
        gru_out1 = gru_out1.view(-1,self.hidden_size)
        gru_out2 = gru_out2.view(-1,self.hidden_size)
        gru_out3 = gru_out3.view(-1,self.hidden_size)
        back = self.back_fc(train_back_data)
        xt = (gru_out1,gru_out2,gru_out3,back)
        xt = F.relu(self.skip_gru_transformation(torch.cat(xt,dim=1)))
        return xt, gru_hid.detach()


