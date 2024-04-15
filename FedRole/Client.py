from FedModel.HGPrediction import *
from FedModel.SpatialModel import *
from FedModel.TemporalModel import *
from FedModel.STCModel import *
from FedModel.STFModel import *
import copy
from FedUtils.utils_fed import *


class Warehouse_Client_Model(torch.nn.Module):
    def __init__(self,warehouse_id):
        
        super(Warehouse_Client_Model,self).__init__()

        self.wareshouse_id = warehouse_id
        self.Local_Temporal_Model = None
        self.Local_STCModel = None
        self.Local_STFModel = None
        self.HGPrediction_lin_dict_out_hid_Warehouse = None
        self.HGPrediction_lin_dict_out_Warehouse = None
        self.HGPrediction_lin_dict_out_hid_pack = None
        self.HGPrediction_lin_dict_out_pack = None
        self.data = {}
        self.Coordinator = None
        self.skip_gru_hidden = {}


    def init_data(self,BaseDir = 'FedData/ClientData/'):
        
        temporal_data = get_node_temporal_data('Warehouse_{}'.format(self.wareshouse_id),BaseDir)
        route_list,warehouse_time,pack_time = get_warehouse_node_data('Warehouse_{}'.format(self.wareshouse_id),BaseDir)

        self.data['temporal_data'] = temporal_data
        self.data['route_list'] = route_list
        self.data['warehouse_time'] = warehouse_time
        self.data['pack_time'] = pack_time
        
    def set_coordinator(self,
                        coordinator):
        self.Coordinator = coordinator

    def forward(self,u_w_s,t,if_first=False):

        u_s = u_w_s.view(1,-1)

        train_temporal_data = self.data['temporal_data'][t]['train_temporal_data'].view(1,-1)
        train_back_data = self.data['temporal_data'][t]['train_back_data'].view(1,-1)
        if_first=if_first
        init_spatial_features = u_s
        
        u_t,gru_hid = self.Local_Temporal_Model(train_temporal_data,
               train_back_data,
               self.Coordinator.now_t,
               self.skip_gru_hidden,
               if_first,
               init_spatial_features)
        self.skip_gru_hidden[self.Coordinator.now_t] = gru_hid
        yield u_t

        

        u_st = self.Local_STCModel(u_s,u_t)
        u_s = self.Local_STFModel(u_s,u_t)
        u_all = torch.cat((u_s,u_t,u_st),dim=1)



        

        downstream_info = self.Coordinator.get_downstram_information(self.data['route_list'])

        u_all = u_all.expand(downstream_info.shape[0],-1)
        u_in = torch.cat((u_all,downstream_info),dim=1)

        w_time = self.HGPrediction_lin_dict_out_Warehouse(F.relu(self.HGPrediction_lin_dict_out_hid_Warehouse(u_in)))
        p_time = self.HGPrediction_lin_dict_out_pack(F.relu(self.HGPrediction_lin_dict_out_hid_pack(u_in)))

        yield w_time,p_time
        


class SortingCenter_Client_Model(torch.nn.Module):
    def __init__(self,sortingcenter_id):
        
        super(SortingCenter_Client_Model,self).__init__()

        self.sortingcenter_id = sortingcenter_id
        self.Local_Temporal_Model = None
        self.Local_STCModel = None
        self.Local_STFModel = None
        self.HGPrediction_lin_dict_out_SortingCenter = None
        self.HGPrediction_lin_dict_out_hid_SortingCenter = None
        self.data = {}
        self.Coordinator = None
        self.skip_gru_hidden = {}

    def init_data(self,BaseDir = 'FedData/ClientData/'):
        
        temporal_data = get_node_temporal_data('SortingCenter_{}'.format(self.sortingcenter_id),BaseDir)
        route_list,sortingcenter_time = get_sortingcenter_node_data('SortingCenter_{}'.format(self.sortingcenter_id),BaseDir)

        self.data['temporal_data'] = temporal_data
        self.data['route_list'] = route_list
        self.data['sortingcenter_time'] = sortingcenter_time
        


    def set_coordinator(self,
                        coordinator):
        self.Coordinator = coordinator

    def forward(self,u_w_s,t,if_first=False):

        u_s = u_w_s.view(1,-1)

        train_temporal_data = self.data['temporal_data'][t]['train_temporal_data'].view(1,-1)
        train_back_data = self.data['temporal_data'][t]['train_back_data'].view(1,-1)
        if_first=if_first
        init_spatial_features = u_s
        
        u_t,gru_hid = self.Local_Temporal_Model(train_temporal_data,
               train_back_data,
               self.Coordinator.now_t,
               self.skip_gru_hidden,
               if_first,
               init_spatial_features)
        self.skip_gru_hidden[self.Coordinator.now_t] = gru_hid
        yield u_t

        

        u_st = self.Local_STCModel(u_s,u_t)
        u_s = self.Local_STFModel(u_s,u_t)
        u_all = torch.cat((u_s,u_t,u_st),dim=1)

        context_information = self.Coordinator.get_context_information(self.data['route_list'],self.sortingcenter_id)

        u_all = u_all.expand(context_information.shape[0],-1)
        u_in = torch.cat((u_all,context_information),dim=1)

        s_time = self.HGPrediction_lin_dict_out_SortingCenter(F.relu(self.HGPrediction_lin_dict_out_hid_SortingCenter(u_in)))
        

        yield s_time
        




        




        





