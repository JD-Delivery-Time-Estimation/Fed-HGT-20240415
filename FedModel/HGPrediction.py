import torch
from FedModel.Attention_Model import Attention, S_Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HGTConv
import copy

class HeterogeneousGraphPrediction(torch.nn.Module):
    def __init__(self, 
                data,
                hidden_channels=32,
                ):
        super(HeterogeneousGraphPrediction,self).__init__()
        self.hidden_size = hidden_channels

        self.fc_down = Linear(4*hidden_channels,4*hidden_channels)
        self.fc_context = Linear(4*hidden_channels,4*hidden_channels)
        self.lin_dict_out_hid = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict_out_hid[node_type] = Linear(8*hidden_channels, 8*hidden_channels)
        self.lin_dict_out = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict_out[node_type] = Linear(-1, 1)
        self.lin_dict_out_hid['pack'] = Linear(8*hidden_channels,8*hidden_channels)
        self.lin_dict_out['pack'] = Linear(8*hidden_channels,1)
        self.communicate_client = {}

    def forward(self,
                xs,
                xt,
                xst,
                mask_in,
                cmask,
                dmask,
                mask_pack_in):
        
        communicate_client = {}
        x_all = {
            'Warehouse':torch.cat((xs['Warehouse'],xt['Warehouse'],xst['Warehouse']),dim=1),
            'SortingCenter':torch.cat((xs['SortingCenter'],xt['SortingCenter'],xst['SortingCenter']),dim=1)
        }

        x_warehouse = torch.matmul(mask_in['Warehouse'],x_all['Warehouse'])

        x_context = self.fc_context(x_all['SortingCenter'])
        x_down = self.fc_down(x_all['SortingCenter'])


        x_context = torch.matmul(cmask,x_context)
        x_down = torch.matmul(dmask,x_down)


        # communicate with client
        
        communicate_client['x_down_to_client'] = x_down
        communicate_client['x_context_to_client'] = x_context
        


        x_out1 = torch.cat((x_warehouse,x_down),dim=1)
        num_sorting_center = x_context.shape[1]
        num_routes = x_context.shape[0]
        x_all_sortingcenter = x_all['SortingCenter'].view(-1,num_sorting_center,4*self.hidden_size)
        x_all_sortingcenter = x_all_sortingcenter.expand(num_routes,num_sorting_center,4*self.hidden_size)
        x_out3 = torch.cat((x_all_sortingcenter,x_context),dim=2)

        x_out2 = x_out1
        mask_in['SortingCenter'] = mask_in['SortingCenter'].view(num_routes,num_sorting_center,-1)
        # x_out3 = torch.matmul(mask_in['SortingCenter'],x_out3)
        x_out3 = x_out3 * mask_in['SortingCenter']
        x_out3 = torch.sum(x_out3,dim=1)
        
        x_out = {
            'Warehouse':x_out1,
            'pack':x_out2,
            'SortingCenter':x_out3
        }
        f_out = {}

        for i in x_out:
            f_out[i] = self.lin_dict_out_hid[i](x_out[i])
            f_out[i] = F.relu(f_out[i])
            f_out[i] = self.lin_dict_out[i](f_out[i])
        
        


        return f_out,communicate_client



        


