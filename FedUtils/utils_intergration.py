

device_type = 'cuda:0'


import torch
class graph_data:
    def __init__(self,store,sort,edge_d,device = device_type):
        self.node_types = ['Warehouse','SortingCenter']
        self.edge_types = [('Warehouse','to','SortingCenter'),('SortingCenter','to','Warehouse'),('SortingCenter','to','SortingCenter')]
        self.meta_data = (self.node_types,self.edge_types)

        self.x_dict = {}
        
        self.x_dict['Warehouse'] = torch.from_numpy(store).to(torch.float32).view(30,-1).to(device)
        self.x_dict['SortingCenter'] = torch.from_numpy(sort).to(torch.float32).view(44,-1).to(device)

        self.edge_index_dict = {}
        self.edge_index_dict[('Warehouse','to','SortingCenter')] = torch.from_numpy(edge_d[('Warehouse','to','SortingCenter')]).to(torch.long).view(2,-1).to(device)
        self.edge_index_dict[('SortingCenter','to','Warehouse')] = torch.from_numpy(edge_d[('SortingCenter','to','Warehouse')]).to(torch.long).view(2,-1).to(device)
        self.edge_index_dict[('SortingCenter','to','SortingCenter')] = torch.from_numpy(edge_d[('SortingCenter','to','SortingCenter')]).to(torch.long).view(2,-1).to(device)


    def metadata(self):
        return self.meta_data

def get_graph(BaseDir='FedData/ServerData/'):
    
    import pandas as pd
    import numpy as np
    df_graph = pd.read_csv(BaseDir + 'Spatial_Graph_Edge.csv')
    store = np.eye(30,30)
    sort = np.eye(44)
    keyl = [('Warehouse','to','SortingCenter'),('SortingCenter','to','Warehouse'),('SortingCenter','to','SortingCenter')]
    edge_d = {}
    for i in keyl:
        edge_d[i] = [[],[]]
    for i in df_graph.iloc:
        key = i['relation_type'].split('_')
        key1 = tuple(key)
        key2 = tuple(key[::-1])
        edge_d[key1][0].append(i['source_id'])
        edge_d[key1][1].append(i['target_id'])
        edge_d[key2][1].append(i['source_id'])
        edge_d[key2][0].append(i['target_id'])
        
    for i in edge_d:
        edge_d[i] = np.array(edge_d[i])
    
    graph = graph_data(store,sort,edge_d)

    return graph


    

def get_ground_truth_time(BaseDir='FedData/ServerData/'):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(BaseDir + 'Y_Delivery_Time.csv')
    y_all_t = np.zeros((240,312))
    y_wh_t = np.zeros((240,312))
    y_sort_t = np.zeros((240,312))
    y_pack_t = np.zeros((240,312))
  

    def timeindex(s):
        return int(s)

    for i in df.iloc:
        time_id = timeindex(i['time'])
        rid = i['route_id']
        rid = int(rid)
        
        t0 = i['full_time']
        t1 = i['Warehouse_time']
        t2 = i['pack_time']
        t3 = i['SortingCenter_time']
        y_all_t[time_id][rid] = t0
        y_wh_t[time_id][rid] = t1
        y_sort_t[time_id][rid] = t3
        y_pack_t[time_id][rid] = t2
    
    


    y_all_t = y_all_t.tolist()
    y_wh_t = y_wh_t.tolist()
    y_sort_t = y_sort_t.tolist()
    y_pack_t = y_pack_t.tolist()

    return y_all_t,y_wh_t ,y_sort_t ,y_pack_t
    


def get_route_mask(BaseDir='FedData/ServerData/'):
    import pandas as pd
    import numpy as np
    import torch
    df = pd.read_csv(BaseDir + 'Spatial_Delivery_Route.csv')
    wh_mask = np.zeros((312,30))
    sort_mask = np.zeros((312,44))
    wh_pack_mask = np.zeros((312,30))
    sort_pack_mask = np.zeros((312,44))
    t = 0
    for i in df.iloc:
        wid = i['Warehouse_id']
        sl = []
        for j in range(5):
            if i['{}_SortingCenter_id'.format(j+1)] != -1:
                sl.append(i['{}_SortingCenter_id'.format(j+1)])

        wh_mask[t][wid] = 1
        for j in sl:
            sort_mask[t][j] = 1
        wh_pack_mask[t][wid] = 1
        sort_pack_mask[t][sl[0]] = 1
        t += 1
    wh_mask = torch.from_numpy(wh_mask)
    sort_mask = torch.from_numpy(sort_mask)
    wh_pack_mask = torch.from_numpy(wh_pack_mask)
    sort_pack_mask = torch.from_numpy(sort_pack_mask)

    return wh_mask,sort_mask,wh_pack_mask,sort_pack_mask


def get_context_mask(BaseDir='FedData/ServerData/'):
    import pandas as pd
    import numpy as np
    import torch

    df1 = pd.read_csv(BaseDir + 'SortingCenter_Rank_Score.csv')
    P_n = df1['SortingCenter_Rank_Score'].tolist()
    def contextual_mask(s_list,P_n):
        cmask = np.zeros((312,44,44))
        p = 0
        for i in s_list:
            n = len(i)
            for j in range(n):
                t = []
                for k in range(n):
                    if j != k:
                        t.append(P_n[i[k]]*(np.e**((3-np.abs(j-k))/10)))
                t = np.array(t)
                t = t/np.sum(t)
                cnt = 0
                for k in range(n):
                    if j!=k:
                        cmask[p][i[j]][i[k]] = t[cnt]
                        cnt+=1
            p += 1
        return cmask
    def downstream_mask(w_list,s_list,P_n):
        N = len(w_list)
        dmask = np.zeros((N,44))
        for i in range(N):
            t = []
            s = 0
            for j in s_list[i]:
                t.append(P_n[j]* (np.e**((4-s)/10)) )
                s += 1
            t = np.array(t)
            t = t/np.sum(t)
            
            for j,score in zip(s_list[i],t):
                dmask[i][j] = score
        return dmask
    df = pd.read_csv(BaseDir + 'Spatial_Delivery_Route.csv')
    w_list = []
    s_list = []

    for i in df.iloc:
        wid = i['Warehouse_id']
        sl = []
        for j in range(5):
            if i['{}_SortingCenter_id'.format(j+1)] != -1:
                sl.append(i['{}_SortingCenter_id'.format(j+1)])
        w_list.append(wid)
        s_list.append(sl)
    dmask = downstream_mask(w_list,s_list,P_n)
    cmask = contextual_mask(s_list,P_n)
    return dmask,cmask

def get_train_data(BaseDir='FedData/ServerData/'):
    y_all_t,y_wh_t ,y_sort_t ,y_pack_t = get_ground_truth_time(BaseDir)
    wh_mask,sort_mask,wh_pack_mask,sort_pack_mask = get_route_mask(BaseDir)
    downstream_mask,context_mask = get_context_mask(BaseDir)
    graph = get_graph()
    return  y_all_t,y_wh_t ,y_sort_t ,y_pack_t \
            ,wh_mask,sort_mask,wh_pack_mask,sort_pack_mask\
            ,downstream_mask,context_mask\
            ,graph


def get_y_time(BaseDir = 'FedData/ServerData/'):
    y_all_t,y_wh_t ,y_sort_t ,y_pack_t = get_ground_truth_time(BaseDir)

    y_all_t_all = []
    y_wh_t_all = []
    y_sort_t_all = []
    y_pack_t_all = []
    for i in y_all_t:
        for j in i:
            y_all_t_all.append(j)
    for i in y_wh_t:
        for j in i:
            y_wh_t_all.append(j)
    for i in y_sort_t:
        for j in i:
            y_sort_t_all.append(j)
    for i in y_pack_t:
        for j in i:
            y_pack_t_all.append(j)
    
    return y_all_t_all,y_wh_t_all,y_pack_t_all,y_sort_t_all
    