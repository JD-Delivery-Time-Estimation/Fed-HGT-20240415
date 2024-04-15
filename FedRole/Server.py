from FedModel.HGPrediction import *
from FedModel.SpatialModel import *
from FedModel.TemporalModel import *
from FedModel.STCModel import *
from FedModel.STFModel import *
import copy
from FedUtils.utils_fed import *
import numpy as np
from FedRole.Client import *
from tqdm import tqdm



class Server_Model(torch.nn.Module):
    def __init__(self,
                 data,
                 hidden_size = 64,
                 num_heads = 4,
                 num_layers = 4
                ):
        super(Server_Model,self).__init__()

        self.Coordinator = None
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.Spatail_Model = HeterogeneousSpatialGNN(data,hidden_channels=self.hidden_size,num_heads=self.num_heads,num_layers=self.num_layers)

        self.Temporal_Model_Warehouse = IntialedGRUwithBackgroundInformation(hidden_channels=self.hidden_size)
        self.Temporal_Model_SortingCenter = IntialedGRUwithBackgroundInformation(hidden_channels=self.hidden_size)

        self.STCModel_Warehouse = SpatialTemporalCorrelationModel_Warehouse(hidden_channels=self.hidden_size)
        self.STCModel_SortingCenter = SpatialTemporalCorrelationModel_SortingCenter(hidden_channels=self.hidden_size)
        self.STFModel_Warehouse = SpatialTemporalFusionModel_Warehouse(hidden_channels=self.hidden_size)
        self.STFModel_SortingCenter = SpatialTemporalFusionModel_SortingCenter(hidden_channels=self.hidden_size)

        self.HGPrediction = HeterogeneousGraphPrediction(data,hidden_channels=self.hidden_size)

    def set_coordinator(self,
                        coordinator):
        self.Coordinator = coordinator
    
    def forward(self, x_dict, edge_index_dict,mask_in,cmask,dmask,mask_pack_in):
        
        # send parameters to client
        Temporal_Model_Warehouse_send = copy.deepcopy(self.Temporal_Model_Warehouse)
        Temporal_Model_SortingCenter_send = copy.deepcopy(self.Temporal_Model_SortingCenter)
        yield Temporal_Model_Warehouse_send, Temporal_Model_SortingCenter_send

        STCModel_Warehouse_send = copy.deepcopy(self.STCModel_Warehouse)
        STCModel_SortingCenter_send = copy.deepcopy(self.STCModel_SortingCenter)
        yield STCModel_Warehouse_send, STCModel_SortingCenter_send

        STFModel_Warehouse_send = copy.deepcopy(self.STFModel_Warehouse)
        STFModel_SortingCenter_send = copy.deepcopy(self.STFModel_SortingCenter)
        yield STFModel_Warehouse_send, STFModel_SortingCenter_send

        HGPrediction_lin_dict_out_Warehouse = copy.deepcopy(self.HGPrediction.lin_dict_out['Warehouse'])
        HGPrediction_lin_dict_out_pack = copy.deepcopy(self.HGPrediction.lin_dict_out['pack'])
        HGPrediction_lin_dict_out_SortingCenter = copy.deepcopy(self.HGPrediction.lin_dict_out['SortingCenter'])
        HGPrediction_lin_dict_out_hid_Warehouse = copy.deepcopy(self.HGPrediction.lin_dict_out_hid['Warehouse'])
        HGPrediction_lin_dict_out_hid_pack = copy.deepcopy(self.HGPrediction.lin_dict_out_hid['pack'])
        HGPrediction_lin_dict_out_hid_SortingCenter = copy.deepcopy(self.HGPrediction.lin_dict_out_hid['SortingCenter'])
        yield HGPrediction_lin_dict_out_Warehouse,HGPrediction_lin_dict_out_hid_Warehouse
        yield HGPrediction_lin_dict_out_pack,HGPrediction_lin_dict_out_hid_pack
        yield HGPrediction_lin_dict_out_SortingCenter,HGPrediction_lin_dict_out_hid_SortingCenter

        out = self.Spatail_Model(x_dict,edge_index_dict)
        xs = out # X_s
        # send aggreate  spatial representation
        yield xs
       
        xt = self.Coordinator.send_to_server_temproal_embedding()
        xst_w = self.STCModel_Warehouse(xs['Warehouse'],xt['Warehouse'])
        xst_s = self.STCModel_SortingCenter(xs['SortingCenter'],xt['SortingCenter'])
        xst = {
            'Warehouse':xst_w,
            'SortingCenter':xst_s
        }
        # xst 
        xup_w = self.STFModel_Warehouse(xs['Warehouse'],xt['Warehouse'])
        xup_s = self.STFModel_SortingCenter(xs['SortingCenter'],xt['SortingCenter'])
        xs['Warehouse'] = xup_w
        xs['SortingCenter'] = xup_s
        f_out,communicate_client = self.HGPrediction(xs,
                xt,
                xst,
                mask_in,
                cmask,
                dmask,
                mask_pack_in)
        yield communicate_client
        yield f_out
        
        


        

        





        



        
