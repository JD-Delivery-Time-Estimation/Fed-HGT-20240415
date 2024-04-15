from   FedUtils.utils_intergration import *
device = 'cuda:0'
def mycopy(tensor):
    return torch.from_numpy(tensor.cpu().detach().numpy()).to(tensor.device)

def get_server_train_data(BaseDir='FedData/ServerData/'):
    y_all_t,y_wh_t ,y_sort_t ,y_pack_t = get_ground_truth_time(BaseDir)
    wh_mask,sort_mask,wh_pack_mask,sort_pack_mask = get_route_mask(BaseDir)
    downstream_mask,context_mask = get_context_mask(BaseDir)
    graph = get_graph()
    return y_all_t,y_wh_t ,y_sort_t ,y_pack_t \
            ,wh_mask,sort_mask,wh_pack_mask,sort_pack_mask\
            ,downstream_mask,context_mask\
            ,graph
def get_node_temporal_data(NodeName,BaseDir):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(BaseDir + NodeName + '_Temporal_Data.csv')
    
    def timeindex(s):
        return int(s)

    train_data = np.zeros((240,1,4))
    train_data_background = np.zeros((240,1,3))

    for i in df.iloc:
        time_id = timeindex(i['time'])
        for j in range(4):
            train_data[time_id][0][j] = i['xt_{}'.format(j)]
        for j in range(3):
            train_data_background[time_id][0][j] = i['xb_{}'.format(j)]
    temporal_data = []
    train_data = torch.from_numpy(train_data).to(device).to(torch.float32)
    train_data_background = torch.from_numpy(train_data_background).to(device).to(torch.float32)
    for i in range(240):
        train_temporal_data = train_data[i].view(1,4)
        train_back_data = train_data_background[i].view(1,3)
        temporal_data.append({
            'train_temporal_data':train_temporal_data,
            'train_back_data':train_back_data
        })
    return temporal_data


def get_client_train_data(BaseDir='FedData/ClientData/'):
    def get_temporal_data(NodeName,BaseDir=BaseDir):
        import pandas as pd
        import numpy as np
        df = pd.read_csv(BaseDir + NodeName + '_Temporal_Data.csv')
        

        def timeindex(s):
            return int(s)

        train_data = np.zeros((240,1,4))
        train_data_background = np.zeros((240,1,3))

        for i in df.iloc:
            time_id = timeindex(i['time'])
            for j in range(4):
                train_data[time_id][0][j] = i['xt_{}'.format(j)]
            for j in range(3):
                train_data_background[time_id][0][j] = i['xb_{}'.format(j)]
           
        
        


        return train_data,train_data_background
    client_data = {

    }
    for i in range(30):
        name = 'Warehouse_' + str(i)
        train_data,train_data_background = get_temporal_data(name)
        client_data[name] = {
            'temporal_data':train_data,
            'background_data':train_data_background
        }
    for i in range(44):
        name = 'SortingCenter_' + str(i)
        train_data,train_data_background = get_temporal_data(name)
        client_data[name] = {
            'temporal_data':train_data,
            'background_data':train_data_background
        }
    return client_data

def get_warehouse_node_data(NodeName,BaseDir):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(BaseDir + NodeName + '_Y_Delivery_Time.csv')
    route_list = sorted(list(set(df['route_id'].values)))
    warehouse_time = []
    pack_time = []
    for t in range(240):
        warehouse_time.append([])
        pack_time.append([])
    for i in df.iloc:
        warehouse_time[int(i['time'])].append(i['Warehouse_time'])
        pack_time[int(i['time'])].append(i['pack_time'])
    for i in range(240):
        warehouse_time[i] = torch.from_numpy(np.array(warehouse_time[i])).to(device).to(torch.float32).view(-1,1)
        pack_time[i] = torch.from_numpy(np.array(pack_time[i])).to(device).to(torch.float32).view(-1,1)
    return route_list,warehouse_time,pack_time

def get_sortingcenter_node_data(NodeName,BaseDir):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(BaseDir + NodeName + '_Y_Delivery_Time.csv')
    route_list = sorted(list(set(df['route_id'].values)))
    sortingcenter_time = []
    
    for t in range(240):
        sortingcenter_time.append([])
        
    for i in df.iloc:
        sortingcenter_time[int(i['time'])].append(i['SortingCneter_time'])
       
    for i in range(240):
        sortingcenter_time[i] = torch.from_numpy(np.array(sortingcenter_time[i])).to(device).to(torch.float32).view(-1,1)
        
    return route_list,sortingcenter_time

def module_mapping():
    return {'Temporal_Model_Warehouse.gru_in_fc.weight': ('Warehouse',
            'Local_Temporal_Model.gru_in_fc.weight'),
            'Temporal_Model_Warehouse.gru_in_fc.bias': ('Warehouse',
            'Local_Temporal_Model.gru_in_fc.bias'),
            'Temporal_Model_Warehouse.gru_init_fc.weight': ('Warehouse',
            'Local_Temporal_Model.gru_init_fc.weight'),
            'Temporal_Model_Warehouse.gru_init_fc.bias': ('Warehouse',
            'Local_Temporal_Model.gru_init_fc.bias'),
            'Temporal_Model_Warehouse.gru.weight_ih_l0': ('Warehouse',
            'Local_Temporal_Model.gru.weight_ih_l0'),
            'Temporal_Model_Warehouse.gru.weight_hh_l0': ('Warehouse',
            'Local_Temporal_Model.gru.weight_hh_l0'),
            'Temporal_Model_Warehouse.gru.bias_ih_l0': ('Warehouse',
            'Local_Temporal_Model.gru.bias_ih_l0'),
            'Temporal_Model_Warehouse.gru.bias_hh_l0': ('Warehouse',
            'Local_Temporal_Model.gru.bias_hh_l0'),
            'Temporal_Model_Warehouse.back_fc.weight': ('Warehouse',
            'Local_Temporal_Model.back_fc.weight'),
            'Temporal_Model_Warehouse.back_fc.bias': ('Warehouse',
            'Local_Temporal_Model.back_fc.bias'),
            'STCModel_Warehouse.att_w.fc_Q.weight': ('Warehouse',
            'Local_STCModel.att_w.fc_Q.weight'),
            'STCModel_Warehouse.att_w.fc_Q.bias': ('Warehouse',
            'Local_STCModel.att_w.fc_Q.bias'),
            'STCModel_Warehouse.att_w.fc_K.weight': ('Warehouse',
            'Local_STCModel.att_w.fc_K.weight'),
            'STCModel_Warehouse.att_w.fc_K.bias': ('Warehouse',
            'Local_STCModel.att_w.fc_K.bias'),
            'STCModel_Warehouse.att_w.fc_V.weight': ('Warehouse',
            'Local_STCModel.att_w.fc_V.weight'),
            'STCModel_Warehouse.att_w.fc_V.bias': ('Warehouse',
            'Local_STCModel.att_w.fc_V.bias'),
            'STCModel_Warehouse.att_w.fc.weight': ('Warehouse',
            'Local_STCModel.att_w.fc.weight'),
            'STCModel_Warehouse.att_w.fc.bias': ('Warehouse',
            'Local_STCModel.att_w.fc.bias'),
            'STFModel_Warehouse.att_w_up.fc_Q.weight': ('Warehouse',
            'Local_STFModel.att_w_up.fc_Q.weight'),
            'STFModel_Warehouse.att_w_up.fc_Q.bias': ('Warehouse',
            'Local_STFModel.att_w_up.fc_Q.bias'),
            'STFModel_Warehouse.att_w_up.fc_K.weight': ('Warehouse',
            'Local_STFModel.att_w_up.fc_K.weight'),
            'STFModel_Warehouse.att_w_up.fc_K.bias': ('Warehouse',
            'Local_STFModel.att_w_up.fc_K.bias'),
            'STFModel_Warehouse.att_w_up.fc_V.weight': ('Warehouse',
            'Local_STFModel.att_w_up.fc_V.weight'),
            'STFModel_Warehouse.att_w_up.fc_V.bias': ('Warehouse',
            'Local_STFModel.att_w_up.fc_V.bias'),
            'STFModel_Warehouse.att_w_up.fc_s.weight': ('Warehouse',
            'Local_STFModel.att_w_up.fc_s.weight'),
            'STFModel_Warehouse.att_w_up.fc_s.bias': ('Warehouse',
            'Local_STFModel.att_w_up.fc_s.bias'),
            'STFModel_Warehouse.att_w_up.fc.weight': ('Warehouse',
            'Local_STFModel.att_w_up.fc.weight'),
            'STFModel_Warehouse.att_w_up.fc.bias': ('Warehouse',
            'Local_STFModel.att_w_up.fc.bias'),
            'STFModel_Warehouse.att_w_up.layer_norm.weight': ('Warehouse',
            'Local_STFModel.att_w_up.layer_norm.weight'),
            'STFModel_Warehouse.att_w_up.layer_norm.bias': ('Warehouse',
            'Local_STFModel.att_w_up.layer_norm.bias'),
            'HGPrediction.lin_dict_out_hid.Warehouse.weight': ('Warehouse',
            'HGPrediction_lin_dict_out_hid_Warehouse.weight'),
            'HGPrediction.lin_dict_out_hid.Warehouse.bias': ('Warehouse',
            'HGPrediction_lin_dict_out_hid_Warehouse.bias'),
            'HGPrediction.lin_dict_out.Warehouse.weight': ('Warehouse',
            'HGPrediction_lin_dict_out_Warehouse.weight'),
            'HGPrediction.lin_dict_out.Warehouse.bias': ('Warehouse',
            'HGPrediction_lin_dict_out_Warehouse.bias'),
            'HGPrediction.lin_dict_out_hid.pack.weight': ('Warehouse',
            'HGPrediction_lin_dict_out_hid_pack.weight'),
            'HGPrediction.lin_dict_out_hid.pack.bias': ('Warehouse',
            'HGPrediction_lin_dict_out_hid_pack.bias'),
            'HGPrediction.lin_dict_out.pack.weight': ('Warehouse',
            'HGPrediction_lin_dict_out_pack.weight'),
            'HGPrediction.lin_dict_out.pack.bias': ('Warehouse',
            'HGPrediction_lin_dict_out_pack.bias'),
            'Temporal_Model_SortingCenter.gru_in_fc.weight': ('SortingCenter',
            'Local_Temporal_Model.gru_in_fc.weight'),
            'Temporal_Model_SortingCenter.gru_in_fc.bias': ('SortingCenter',
            'Local_Temporal_Model.gru_in_fc.bias'),
            'Temporal_Model_SortingCenter.gru_init_fc.weight': ('SortingCenter',
            'Local_Temporal_Model.gru_init_fc.weight'),
            'Temporal_Model_SortingCenter.gru_init_fc.bias': ('SortingCenter',
            'Local_Temporal_Model.gru_init_fc.bias'),
            'Temporal_Model_SortingCenter.gru.weight_ih_l0': ('SortingCenter',
            'Local_Temporal_Model.gru.weight_ih_l0'),
            'Temporal_Model_SortingCenter.gru.weight_hh_l0': ('SortingCenter',
            'Local_Temporal_Model.gru.weight_hh_l0'),
            'Temporal_Model_SortingCenter.gru.bias_ih_l0': ('SortingCenter',
            'Local_Temporal_Model.gru.bias_ih_l0'),
            'Temporal_Model_SortingCenter.gru.bias_hh_l0': ('SortingCenter',
            'Local_Temporal_Model.gru.bias_hh_l0'),
            'Temporal_Model_SortingCenter.back_fc.weight': ('SortingCenter',
            'Local_Temporal_Model.back_fc.weight'),
            'Temporal_Model_SortingCenter.back_fc.bias': ('SortingCenter',
            'Local_Temporal_Model.back_fc.bias'),
            'STCModel_SortingCenter.att_s.fc_Q.weight': ('SortingCenter',
            'Local_STCModel.att_s.fc_Q.weight'),
            'STCModel_SortingCenter.att_s.fc_Q.bias': ('SortingCenter',
            'Local_STCModel.att_s.fc_Q.bias'),
            'STCModel_SortingCenter.att_s.fc_K.weight': ('SortingCenter',
            'Local_STCModel.att_s.fc_K.weight'),
            'STCModel_SortingCenter.att_s.fc_K.bias': ('SortingCenter',
            'Local_STCModel.att_s.fc_K.bias'),
            'STCModel_SortingCenter.att_s.fc_V.weight': ('SortingCenter',
            'Local_STCModel.att_s.fc_V.weight'),
            'STCModel_SortingCenter.att_s.fc_V.bias': ('SortingCenter',
            'Local_STCModel.att_s.fc_V.bias'),
            'STCModel_SortingCenter.att_s.fc.weight': ('SortingCenter',
            'Local_STCModel.att_s.fc.weight'),
            'STCModel_SortingCenter.att_s.fc.bias': ('SortingCenter',
            'Local_STCModel.att_s.fc.bias'),
            'STFModel_SortingCenter.att_s_up.fc_Q.weight': ('SortingCenter',
            'Local_STFModel.att_s_up.fc_Q.weight'),
            'STFModel_SortingCenter.att_s_up.fc_Q.bias': ('SortingCenter',
            'Local_STFModel.att_s_up.fc_Q.bias'),
            'STFModel_SortingCenter.att_s_up.fc_K.weight': ('SortingCenter',
            'Local_STFModel.att_s_up.fc_K.weight'),
            'STFModel_SortingCenter.att_s_up.fc_K.bias': ('SortingCenter',
            'Local_STFModel.att_s_up.fc_K.bias'),
            'STFModel_SortingCenter.att_s_up.fc_V.weight': ('SortingCenter',
            'Local_STFModel.att_s_up.fc_V.weight'),
            'STFModel_SortingCenter.att_s_up.fc_V.bias': ('SortingCenter',
            'Local_STFModel.att_s_up.fc_V.bias'),
            'STFModel_SortingCenter.att_s_up.fc_s.weight': ('SortingCenter',
            'Local_STFModel.att_s_up.fc_s.weight'),
            'STFModel_SortingCenter.att_s_up.fc_s.bias': ('SortingCenter',
            'Local_STFModel.att_s_up.fc_s.bias'),
            'STFModel_SortingCenter.att_s_up.fc.weight': ('SortingCenter',
            'Local_STFModel.att_s_up.fc.weight'),
            'STFModel_SortingCenter.att_s_up.fc.bias': ('SortingCenter',
            'Local_STFModel.att_s_up.fc.bias'),
            'STFModel_SortingCenter.att_s_up.layer_norm.weight': ('SortingCenter',
            'Local_STFModel.att_s_up.layer_norm.weight'),
            'STFModel_SortingCenter.att_s_up.layer_norm.bias': ('SortingCenter',
            'Local_STFModel.att_s_up.layer_norm.bias'),
            'HGPrediction.lin_dict_out.SortingCenter.weight': ('SortingCenter',
            'HGPrediction_lin_dict_out_SortingCenter.weight'),
            'HGPrediction.lin_dict_out.SortingCenter.bias': ('SortingCenter',
            'HGPrediction_lin_dict_out_SortingCenter.bias'),
            'HGPrediction.lin_dict_out_hid.SortingCenter.weight': ('SortingCenter',
            'HGPrediction_lin_dict_out_hid_SortingCenter.weight'),
            'HGPrediction.lin_dict_out_hid.SortingCenter.bias': ('SortingCenter',
            'HGPrediction_lin_dict_out_hid_SortingCenter.bias')}