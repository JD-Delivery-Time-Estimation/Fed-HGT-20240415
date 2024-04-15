from FedModel.HGPrediction import *
from FedModel.SpatialModel import *
from FedModel.TemporalModel import *
from FedModel.STCModel import *
from FedModel.STFModel import *
import copy
from FedUtils.Metric_utils import *
from FedUtils.utils_fed import *
import numpy as np
from FedRole.Client import *
from FedRole.Server import *
from tqdm import tqdm
import os
from datetime import datetime


class Coordinator:
    def __init__(self,
                clients=None,
                server=None,
                device='cuda:0',
                beta=0.1,
                time_size=240,
                log_basepath = 'Log/',
                hidden_size = 64,
                num_heads = 4,
                num_layers = 2,
                num_warehouse = 30,
                num_sortingcenter = 44):
        self.num_warehouse = num_warehouse
        self.num_sortingcenter = num_sortingcenter
        self.clients = clients
        self.server = server
        self.server_forward_iter = None
        self.client_forward_iter = {}
        self.global_information = {}
        self.device = device
        self.server_data = {}
        self.ut = {}
        self.optimizer = {}
        self.client_lr = 0.0001
        self.server_lr = 0.0001
        self.lossfunction = torch.nn.MSELoss()
        
        self.N_train = int(time_size * 0.8)
        self.N_test = int(time_size * 0.2)

        self.train_range = list(range(self.N_train))
        self.test_range = list(range(self.N_train,self.N_train+self.N_test))
        
        self.epoch = None


        self.beta =beta
        self.module_mapping_dict = module_mapping()
        self.Logdir = log_basepath + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        os.mkdir(self.Logdir)
        self.Modeldir = self.Logdir + '/Model/'
        os.mkdir(self.Modeldir)
        self.Traindir = self.Logdir + '/train/'
        os.mkdir(self.Traindir)
        self.Testdir = self.Logdir + '/test/'
        os.mkdir(self.Testdir)
        self.gru_hidden_dir = self.Logdir + '/gru_hidden/'
        os.mkdir(self.gru_hidden_dir)

        self.f_server_loss = self.Logdir + '/train/server_loss.txt'
        self.f_warehouse_loss = self.Logdir + '/train/warehouse_loss.txt'
        self.f_pack_loss = self.Logdir + '/train/pack_loss.txt'
        self.f_sortingcenter_loss = self.Logdir + '/train/sortingcenter_loss.txt'
        self.f_metric = self.Logdir + '/train/metric.txt'
        
        self.f_server_loss_train = self.Logdir + '/train/train_server_loss.txt'
        self.f_warehouse_loss_train = self.Logdir + '/train/train_warehouse_loss.txt'
        self.f_sortingcenter_loss_train = self.Logdir + '/train/train_sortingcenter_loss.txt'

        self.server_loss = 0.0
        self.warehouse_loss = 0.0
        self.pack_loss = 0.0
        self.sortingcenter_loss = 0.0
        self.server_metric = None

        self.outall = None
        self.outwh = None
        self.outpack = None
        self.outsort = None


        self.f_server_loss_test = self.Logdir + '/test/server_loss_test.txt'
        self.f_warehouse_loss_test = self.Logdir + '/test/warehouse_loss_test.txt'
        self.f_pack_loss_test = self.Logdir + '/test/pack_loss_test.txt'
        self.f_sortingcenter_loss_test = self.Logdir + '/test/sortingcenter_loss_test.txt'
        self.f_metric_test = self.Logdir + '/test/metric_test.txt'
        
        self.f_server_loss_train_test = self.Logdir + '/test/test_server_loss.txt'
        self.f_warehouse_loss_train_test = self.Logdir + '/test/test_warehouse_loss.txt'
        self.f_sortingcenter_loss_train_test = self.Logdir + '/test/test_sortingcenter_loss.txt'

        self.server_loss_test = 0.0
        self.warehouse_loss_test = 0.0
        self.pack_loss_test = 0.0
        self.sortingcenter_loss_test = 0.0
        self.server_metric_test = None

        self.outall_test = None
        self.outwh_test = None
        self.outpack_test = None
        self.outsort_test = None

        self.hidden_size = hidden_size 
        self.num_heads = num_heads 
        self.num_layers = num_layers 
        if self.server is None:
            self.init_server()
        if clients is None:
            self.init_clients_using_federate_dataset()
        self.now_t = 0

    def load_parameters(self,parameters_path):
        import pickle
        with open(parameters_path,'rb') as f:
            para = pickle.load(f)
        self.server.load_state_dict(para)
    
    def load_gru_hiddens(self,hidden_path):
        import pickle
        for i in range(self.num_sortingcenter):
            with open(hidden_path+'SortingCenter_{}.pkl'.format(i),'rb') as f:
                para = pickle.load(f)
            self.clients['SortingCenter'][i].skip_gru_hidden = para
        for i in range(self.num_warehouse):
            with open(hidden_path+'Warehouse_{}.pkl'.format(i),'rb') as f:
                para = pickle.load(f)
            self.clients['Warehouse'][i].skip_gru_hidden = para

    def train_epoch(self,epoch):
        self.epoch = epoch
        print('--------EPOCH: {}--------'.format(epoch))
        for t in tqdm(self.train_range):
            self.now_t = t
            self.run_at_t(t)
        self.Log_and_Save_Model()
    
    def test(self):
        for t in tqdm(self.test_range):
            self.now_t = t
            self.test_at_t(t)
        self.Log_Test()

    def run_at_t(self,t):
        self.set_server_train()
        self.init_server_iter()
        self.set_clients_parameters()
        self.set_client_train()
        if t == 0:
            self.set_client_optimizer()
            self.set_server_optimizer()

        self.init_client_iter(t)
        self.collect_temporal_embeddings()
        self.set_global_info()
        
        self.server_train(t)
        self.client_train(t)

        self.clients_send_parameters_to_server()
    
    def test_at_t(self,t):
        self.set_server_eval()
        self.set_client_eval()
        self.init_server_iter()
        self.set_clients_parameters()
        self.init_client_iter(t)
        self.collect_temporal_embeddings()
        self.set_global_info()
        self.server_test(t)
        self.client_test(t)
        self.clients_send_parameters_to_server()


    def init_clients_using_federate_dataset(self):
        self.clients = {}
        self.clients['Warehouse'] = []
        self.clients['SortingCenter'] = []
        print("--------INIT WAREHOUSE CLIENTS--------")
        for i in tqdm(range(self.num_warehouse)):
            client = Warehouse_Client_Model(i)
            client.init_data()
            client.set_coordinator(self)
            self.clients['Warehouse'] .append(client)
        print("--------INIT SORTING CENTER CLIENTS--------")
        for i in tqdm(range(self.num_sortingcenter)):
            client = SortingCenter_Client_Model(i)
            client.init_data()
            client.set_coordinator(self)
            self.clients['SortingCenter'] .append(client)
        print("--------INIT CLIENTS FINISHED--------")

    def init_clients(self,warehouse_number,sortingcenter_number):
        self.clients = {}
        self.clients['Warehouse'] = []
        self.clients['SortingCenter'] = []
        print("--------INIT WAREHOUSE CLIENTS--------")
        for i in tqdm(range(warehouse_number)):
            client = Warehouse_Client_Model(i)
            client.init_data()
            client.set_coordinator(self)
            self.clients['Warehouse'] .append(client)
        print("--------INIT SORTING CENTER CLIENTS--------")
        for i in tqdm(range(sortingcenter_number)):
            client = SortingCenter_Client_Model(i)
            client.init_data()
            client.set_coordinator(self)
            self.clients['SortingCenter'] .append(client)
        print("--------INIT CLIENTS FINISHED--------")

    def init_server(self):
        y_all_t,y_wh_t ,y_sort_t ,y_pack_t \
        ,wh_mask,sort_mask,wh_pack_mask,sort_pack_mask\
        ,downstream_mask,context_mask\
        ,graph = get_server_train_data()
        server = Server_Model(data=graph,
                              hidden_size = self.hidden_size,
                              num_heads = self.num_heads,
                              num_layers = self.num_layers)
        device = self.device
        wh_mask = wh_mask.to(device).to(torch.float32)
        sort_mask = sort_mask.to(device).to(torch.float32)
        mask_in = {
            'Warehouse':wh_mask,
            'SortingCenter':sort_mask
        }
        wh_pack_mask = wh_pack_mask.to(device).to(torch.float32)
        sort_pack_mask = sort_pack_mask.to(device).to(torch.float32)
        mask_pack_in = {
            'Warehouse':wh_pack_mask,
            'SortingCenter':sort_pack_mask
        }
        downmask_ins =torch.from_numpy(downstream_mask).to(device).to(torch.float32)
        cmask_ins=torch.from_numpy(context_mask).to(device).to(torch.float32)
        x_dict = graph.x_dict
        edge_index_dict = graph.edge_index_dict
        cmask = cmask_ins
        dmask = downmask_ins

        y_all_ts = [] 
        y_wh_ts = []
        y_sort_ts = []
        y_pack_ts = []
        import numpy as np
        for i in range(240):
            y_all_ts.append(torch.from_numpy(np.array(y_all_t[i])).to(device).to(torch.float32).view(-1,1))
            y_wh_ts.append(torch.from_numpy(np.array(y_wh_t[i])).to(device).to(torch.float32).view(-1,1))
            y_sort_ts.append(torch.from_numpy(np.array(y_sort_t[i])).to(device).to(torch.float32).view(-1,1))
            y_pack_ts.append(torch.from_numpy(np.array(y_pack_t[i])).to(device).to(torch.float32).view(-1,1))

        self.server_data['x_dict'] = x_dict
        self.server_data['edge_index_dict'] = edge_index_dict
        self.server_data['mask_in'] = mask_in
        self.server_data['cmask'] = cmask
        self.server_data['dmask'] = dmask
        self.server_data['mask_pack_in'] = mask_pack_in
        self.server_data['y_all_ts'] = y_all_ts
        self.server_data['y_wh_ts'] = y_wh_ts
        self.server_data['y_sort_ts'] = y_sort_ts
        self.server_data['y_pack_ts'] = y_pack_ts

        import numpy as np
        y_all_t_all,y_wh_t_all,y_pack_t_all,y_sort_t_all = get_y_time()
        y_all_t_all = torch.from_numpy(np.array(y_all_t_all)).to('cpu').to(torch.float32)
        y_wh_t_all = torch.from_numpy(np.array(y_wh_t_all)).to('cpu').to(torch.float32)
        y_pack_t_all = torch.from_numpy(np.array(y_pack_t_all)).to('cpu').to(torch.float32)
        y_sort_t_all = torch.from_numpy(np.array(y_sort_t_all)).to('cpu').to(torch.float32)

        self.server_data['y_all_t_all'] = y_all_t_all
        self.server_data['y_wh_t_all'] = y_wh_t_all
        self.server_data['y_pack_t_all'] = y_pack_t_all
        self.server_data['y_sort_t_all'] = y_sort_t_all

        self.server = server
        server.to(device=self.device)
        server.float()
        server.set_coordinator(self)
        print("--------INIT SERVER FINISHED--------")



        
    def init_server_iter(self):
        x_dict = self.server_data['x_dict']
        edge_index_dict = self.server_data['edge_index_dict'] 
        mask_in = self.server_data['mask_in'] 
        cmask = self.server_data['cmask']  
        dmask = self.server_data['dmask'] 
        mask_pack_in = self.server_data['mask_pack_in'] 
        self.server_forward_iter = self.server(x_dict, edge_index_dict,mask_in,cmask,dmask,mask_pack_in)
    
    def set_clients_parameters(self):
        Temporal_Model_Warehouse_send, Temporal_Model_SortingCenter_send = next(self.server_forward_iter)
        STCModel_Warehouse_send, STCModel_SortingCenter_send = next(self.server_forward_iter)
        STFModel_Warehouse_send, STFModel_SortingCenter_send = next(self.server_forward_iter)
        HGPrediction_lin_dict_out_Warehouse,HGPrediction_lin_dict_out_hid_Warehouse = next(self.server_forward_iter)
        HGPrediction_lin_dict_out_pack,HGPrediction_lin_dict_out_hid_pack = next(self.server_forward_iter)
        HGPrediction_lin_dict_out_SortingCenter,HGPrediction_lin_dict_out_hid_SortingCenter = next(self.server_forward_iter)
        n_warehouse = len(self.clients['Warehouse'])
        n_sortingcenter = len(self.clients['SortingCenter'])
        for i in range(n_warehouse):
            self.clients['Warehouse'][i].Local_Temporal_Model =  Temporal_Model_Warehouse_send
            self.clients['Warehouse'][i].Local_STCModel = copy.deepcopy(STCModel_Warehouse_send)
            self.clients['Warehouse'][i].Local_STFModel = copy.deepcopy(STFModel_Warehouse_send)
            self.clients['Warehouse'][i].HGPrediction_lin_dict_out_hid_Warehouse = copy.deepcopy(HGPrediction_lin_dict_out_hid_Warehouse)
            self.clients['Warehouse'][i].HGPrediction_lin_dict_out_Warehouse = copy.deepcopy(HGPrediction_lin_dict_out_Warehouse)
            self.clients['Warehouse'][i].HGPrediction_lin_dict_out_hid_pack = copy.deepcopy(HGPrediction_lin_dict_out_hid_pack)
            self.clients['Warehouse'][i].HGPrediction_lin_dict_out_pack = copy.deepcopy(HGPrediction_lin_dict_out_pack)
            self.clients['Warehouse'][i].to(device=self.device)
            self.clients['Warehouse'][i].float()
        for i in range(n_sortingcenter):
            self.clients['SortingCenter'][i].Local_Temporal_Model =  Temporal_Model_SortingCenter_send
            self.clients['SortingCenter'][i].Local_STCModel = copy.deepcopy(STCModel_SortingCenter_send)
            self.clients['SortingCenter'][i].Local_STFModel = copy.deepcopy(STFModel_SortingCenter_send)
            self.clients['SortingCenter'][i].HGPrediction_lin_dict_out_SortingCenter = copy.deepcopy(HGPrediction_lin_dict_out_SortingCenter)
            self.clients['SortingCenter'][i].HGPrediction_lin_dict_out_hid_SortingCenter = copy.deepcopy(HGPrediction_lin_dict_out_hid_SortingCenter)
            self.clients['SortingCenter'][i].to(device=self.device)
            self.clients['SortingCenter'][i].float()
    
    def set_client_train(self):
        n_warehouse = len(self.clients['Warehouse'])
        n_sortingcenter = len(self.clients['SortingCenter'])
        for i in range(n_warehouse):
            self.clients['Warehouse'][i].train()
        for i in range(n_sortingcenter):
            self.clients['SortingCenter'][i].train()
    
    def set_client_eval(self):
        n_warehouse = len(self.clients['Warehouse'])
        n_sortingcenter = len(self.clients['SortingCenter'])
        for i in range(n_warehouse):
            self.clients['Warehouse'][i].eval()
        for i in range(n_sortingcenter):
            self.clients['SortingCenter'][i].eval()
    
    def set_server_train(self):
        self.server.train()
    
    def set_server_eval(self):
        self.server.eval()

    def init_client_iter(self,t):
        if_first = False
        if t == 0:
            if_first = True
        if 'Warehouse' in  self.client_forward_iter :
            del  self.client_forward_iter['Warehouse'] 
        if 'SortingCenter' in self.client_forward_iter:
            del self.client_forward_iter['SortingCenter']
        self.client_forward_iter['Warehouse'] = []
        self.client_forward_iter['SortingCenter'] = []
        u_s = next(self.server_forward_iter)
        n_warehouse = len(self.clients['Warehouse'])
        n_sortingcenter = len(self.clients['SortingCenter'])
        for i in range(n_warehouse):
            u_w_s_test_warehouse = mycopy(u_s['Warehouse'][i])
            self.client_forward_iter['Warehouse'].append(self.clients['Warehouse'][i](u_w_s_test_warehouse,t,if_first))
        for i in range(n_sortingcenter):
            u_w_s_test_sortingcenter = mycopy(u_s['SortingCenter'][i])
            self.client_forward_iter['SortingCenter'].append(self.clients['SortingCenter'][i](u_w_s_test_sortingcenter,t,if_first)) 

    def set_global_info(self):
        communicate_client = next(self.server_forward_iter)
        x_down = communicate_client['x_down_to_client']
        x_context = communicate_client['x_context_to_client'] 
        self.global_information['Context_Information'] = mycopy(x_context)
        self.global_information['Downstream_Information'] = mycopy(x_down)

    def set_client_optimizer(self):
        if 'Warehouse' in self.optimizer:
            del self.optimizer['Warehouse']
        if 'SortingCenter' in self.optimizer:
            del self.optimizer['SortingCenter']
        self.optimizer['Warehouse'] = []
        self.optimizer['SortingCenter'] = []
        n_warehouse = len(self.clients['Warehouse'])
        n_sortingcenter = len(self.clients['SortingCenter'])
        for i in range(n_warehouse):
            opt = torch.optim.Adam(self.clients['Warehouse'][i].parameters(),lr=self.client_lr)
            self.optimizer['Warehouse'].append(opt)
        for i in range(n_sortingcenter):
            opt = torch.optim.Adam(self.clients['SortingCenter'][i].parameters(),lr=self.client_lr)
            self.optimizer['SortingCenter'].append(opt)
    
    def set_server_optimizer(self):
        self.optimizer['server'] = torch.optim.Adam(self.server.parameters(),lr=self.server_lr)

    def server_train(self,t):
        if t == 0:
            self.server_loss = 0.0

        f_out = next(self.server_forward_iter)
        out = f_out
        y_wh_t = self.server_data['y_wh_ts'][t]
        y_sort_t = self.server_data['y_sort_ts'][t]
        y_pack_t =  self.server_data['y_pack_ts'][t]
        y_all_t = self.server_data['y_all_ts'][t]
        loss_wh = self.lossfunction(out['Warehouse'],y_wh_t)
        loss_sort = self.lossfunction(out['SortingCenter'],y_sort_t)
        loss_pack = self.lossfunction(out['pack'],y_pack_t)
        loss_all = self.lossfunction(out['Warehouse']+out['SortingCenter']+out['pack'],y_all_t)
        loss = loss_wh + loss_sort + loss_all + loss_pack
        self.optimizer['server'].zero_grad()
        loss.backward()
        self.optimizer['server'].step()
        self.print_data(data="[SERVER] EPOCH {} @ Time {}: Server Warehouse Loss: {}, Server Pack Loss: {}, Server SortingCenter Loss: {},Server Full-link Loss: {}".format(
            self.epoch,
            t,
            float(loss_wh),
            float(loss_pack),
            float(loss_sort),
            float(loss)
            ),
            file=self.f_server_loss_train)
        self.server_loss += float(loss)
        if t == 0:
            self.outall = (out['Warehouse']+out['SortingCenter']+out['pack']).cpu().detach()
            self.outwh = out['Warehouse'].cpu().detach()
            self.outpack = out['pack'].cpu().detach()
            self.outsort = out['SortingCenter'].cpu().detach()
        else:
            self.outall = torch.cat((self.outall,(out['Warehouse']+out['SortingCenter']+out['pack']).cpu().detach()),dim=0)
            self.outwh = torch.cat((self.outwh,out['Warehouse'].cpu().detach()),dim=0)
            self.outpack = torch.cat((self.outpack,out['pack'].cpu().detach()),dim=0)
            self.outsort = torch.cat((self.outsort,out['SortingCenter'].cpu().detach()),dim=0)
    
    def server_test(self,t):
        if t == self.test_range[0]:
            self.server_loss_test = 0.0

        f_out = next(self.server_forward_iter)
        out = f_out
        y_wh_t = self.server_data['y_wh_ts'][t]
        y_sort_t = self.server_data['y_sort_ts'][t]
        y_pack_t =  self.server_data['y_pack_ts'][t]
        y_all_t = self.server_data['y_all_ts'][t]
        loss_wh_test = self.lossfunction(out['Warehouse'],y_wh_t)
        loss_sort_test = self.lossfunction(out['SortingCenter'],y_sort_t)
        loss_pack_test = self.lossfunction(out['pack'],y_pack_t)
        loss_all_test = self.lossfunction(out['Warehouse']+out['SortingCenter']+out['pack'],y_all_t)
        loss_test = loss_wh_test + loss_sort_test + loss_all_test + loss_pack_test
        self.print_data(data="[TEST SERVER] @ Time {}: Server Warehouse Loss: {}, Server Pack Loss: {}, Server SortingCenter Loss: {},Server Full-link Loss: {}".format(
            t,
            float(loss_wh_test),
            float(loss_pack_test),
            float(loss_sort_test),
            float(loss_test)
            ),
            file=self.f_server_loss_train_test)
        self.server_loss_test += float(loss_test)
        if t == self.test_range[0]:
            self.outall_test = (out['Warehouse']+out['SortingCenter']+out['pack']).cpu().detach()
            self.outwh_test = out['Warehouse'].cpu().detach()
            self.outpack_test = out['pack'].cpu().detach()
            self.outsort_test = out['SortingCenter'].cpu().detach()
        else:
            self.outall_test = torch.cat((self.outall_test,(out['Warehouse']+out['SortingCenter']+out['pack']).cpu().detach()),dim=0)
            self.outwh_test = torch.cat((self.outwh_test,out['Warehouse'].cpu().detach()),dim=0)
            self.outpack_test = torch.cat((self.outpack_test,out['pack'].cpu().detach()),dim=0)
            self.outsort_test = torch.cat((self.outsort_test,out['SortingCenter'].cpu().detach()),dim=0)
    
   


    def client_train(self,t):
        n_warehouse = len(self.clients['Warehouse'])
        n_sortingcenter = len(self.clients['SortingCenter'])
        if t == 0: 
            self.warehouse_loss = 0.0
            self.pack_loss = 0.0
            self.sortingcenter_loss = 0.0

        for i in range(n_warehouse):
            w_time,p_time = next(self.client_forward_iter['Warehouse'][i])
            loss_wh = self.lossfunction(w_time,self.clients['Warehouse'][i].data['warehouse_time'][t])
            loss_pack = self.lossfunction(p_time,self.clients['Warehouse'][i].data['pack_time'][t])
            
            self.warehouse_loss += float(loss_wh)
            self.pack_loss += float(loss_pack)

            loss = loss_wh +  loss_pack
            self.optimizer['Warehouse'][i].zero_grad()
            loss.backward()
            self.optimizer['Warehouse'][i].step()
            self.print_data(data="[CLIENT Warehouse ID {}] EPOCH {} @ Time {}: Client Warehouse Loss: {}, Client Pack Loss: {}, Client Loss: {}".format(
                i,
                self.epoch,
                t,
                float(loss_wh),
                float(loss_pack),
                float(loss)
                ),
                file=self.f_warehouse_loss_train)
        for i in range(n_sortingcenter):
            s_time = next(self.client_forward_iter['SortingCenter'][i])
            loss_sort = self.lossfunction(s_time,self.clients['SortingCenter'][i].data['sortingcenter_time'][t])
            loss = loss_sort
            self.sortingcenter_loss += float(loss_sort)
            self.optimizer['SortingCenter'][i].zero_grad()
            loss.backward()
            self.optimizer['SortingCenter'][i].step()
            self.print_data(data="[CLIENT SortingCenter ID {}] EPOCH {} @ Time {}: Client SortingCenter Loss: {}".format(
                i,
                self.epoch,
                t,
                float(loss)
                ),
                file=self.f_sortingcenter_loss_train)
            
    def client_test(self,t):
        n_warehouse = len(self.clients['Warehouse'])
        n_sortingcenter = len(self.clients['SortingCenter'])
        if t == self.test_range[0]: 
            self.warehouse_loss_test = 0.0
            self.pack_loss_test = 0.0
            self.sortingcenter_loss_test = 0.0

        for i in range(n_warehouse):
            w_time,p_time = next(self.client_forward_iter['Warehouse'][i])
            loss_wh_test = self.lossfunction(w_time,self.clients['Warehouse'][i].data['warehouse_time'][t])
            loss_pack_test = self.lossfunction(p_time,self.clients['Warehouse'][i].data['pack_time'][t])
            
            self.warehouse_loss_test += float(loss_wh_test)
            self.pack_loss_test += float(loss_pack_test)

            loss_test = loss_wh_test +  loss_pack_test
            self.print_data(data="[TEST CLIENT Warehouse ID {}] @ Time {}: Client Warehouse Loss: {}, Client Pack Loss: {}, Client Loss: {}".format(
                i,
                t,
                float(loss_wh_test),
                float(loss_pack_test),
                float(loss_test)
                ),
                file=self.f_warehouse_loss_train_test)
        for i in range(n_sortingcenter):
            s_time = next(self.client_forward_iter['SortingCenter'][i])
            loss_sort_test = self.lossfunction(s_time,self.clients['SortingCenter'][i].data['sortingcenter_time'][t])
            loss_test = loss_sort_test
            self.sortingcenter_loss_test += float(loss_sort_test)
            self.print_data(data="[TEST CLIENT SortingCenter ID {}] @ Time {}: Client SortingCenter Loss: {}".format(
                i,
                t,
                float(loss_test)
                ),
                file=self.f_sortingcenter_loss_train_test)
    


    def collect_temporal_embeddings(self):

        n_warehouse = len(self.clients['Warehouse'])
        n_sortingcenter = len(self.clients['SortingCenter'])
        u_t_w = None
        for i in range(n_warehouse):
            if i == 0:
                u_t_w = mycopy(next(self.client_forward_iter['Warehouse'][i]))
            else:
                u_t_w = torch.cat((u_t_w,mycopy(next(self.client_forward_iter['Warehouse'][i]))),dim=0)
        u_t_s = None
        for i in range(n_sortingcenter):
            if i == 0:
                u_t_s = mycopy(next(self.client_forward_iter['SortingCenter'][i]))
            else:
                u_t_s = torch.cat((u_t_s,mycopy(next(self.client_forward_iter['SortingCenter'][i]))),dim=0)
        self.ut = {
            'Warehouse':u_t_w,
            'SortingCenter':u_t_s
        }
        

    def send_to_server_temproal_embedding(self):
        return self.ut
    
    def get_context_information(self,route_list,sorting_center_id):
        return copy.deepcopy(self.global_information['Context_Information'][route_list,sorting_center_id]) 

    def get_downstram_information(self,route_list):
        return copy.deepcopy(self.global_information['Downstream_Information'][route_list])


    def clients_send_parameters_to_server(self):
        n_warehouse = len(self.clients['Warehouse'])
        n_sortingcenter = len(self.clients['SortingCenter'])
        moudles = list(self.server.state_dict().keys())
        new_server_parameters = self.server.state_dict()
        for i in moudles:
            if i not in self.module_mapping_dict:
                continue
            else:
                new_server_parameters[i] = (1-self.beta) * new_server_parameters[i]
                if self.module_mapping_dict[i][0] == 'Warehouse':
                    for j in range(n_warehouse):
                        new_server_parameters[i] += 1/n_warehouse * self.clients['Warehouse'][j].state_dict()[self.module_mapping_dict[i][1]] * self.beta
                if self.module_mapping_dict[i][0] == 'SortingCenter':
                    for j in range(n_sortingcenter):
                        new_server_parameters[i] += 1/n_sortingcenter * self.clients['SortingCenter'][j].state_dict()[self.module_mapping_dict[i][1]] * self.beta
        self.server.load_state_dict(new_server_parameters)
    
    def Log_and_Save_Model(self):
        pres = {
            "all":self.outall,
            "store":self.outwh,
            "pack":self.outpack,
            "sort":self.outsort
        }
        yts =  {
            "all":self.server_data['y_all_t_all'][0:312*self.N_train],
            "store":self.server_data['y_wh_t_all'][0:312*self.N_train],
            "pack":self.server_data['y_pack_t_all'][0:312*self.N_train],
            "sort":self.server_data['y_sort_t_all'][0:312*self.N_train]
        }
        self.server_metric = metric_all(pres,yts)
        self.server_loss = self.server_loss / self.N_train
        self.warehouse_loss = self.warehouse_loss / self.N_train / len(self.clients['Warehouse'])
        self.pack_loss = self.pack_loss / self.N_train / len(self.clients['Warehouse'])
        self.sortingcenter_loss = self.sortingcenter_loss / self.N_train / len(self.clients['SortingCenter'])
        
        print("EPOCH {}: Server Loss:{}".format(self.epoch,self.server_loss))
        print("EPOCH {}: Warehouse Client Loss:{}".format(self.epoch,self.warehouse_loss))
        print("EPOCH {}: Warehouse Client pack Loss:{}".format(self.epoch,self.pack_loss))
        print("EPOCH {}: SortingCenter Client Loss:{}".format(self.epoch,self.sortingcenter_loss))

        self.print_data(data="EPOCH {}: Server Loss:{}".format(self.epoch,self.server_loss),file=self.f_server_loss)
        self.print_data(data="EPOCH {}: Warehouse Client Loss:{}".format(self.epoch,self.warehouse_loss),file=self.f_warehouse_loss)
        self.print_data(data="EPOCH {}: Warehouse Client pack Loss:{}".format(self.epoch,self.pack_loss),file=self.f_pack_loss)
        self.print_data(data="EPOCH {}: SortingCenter Client Loss:{}".format(self.epoch,self.sortingcenter_loss),file=self.f_sortingcenter_loss)
        self.print_data(data="EPOCH {}: Server Mertics:".format(self.epoch)+str(self.server_metric),file=self.f_metric)
        import pickle
        if self.epoch % 2 == 0:
            with open(self.Modeldir + '{}_{}.pkl'.format(self.epoch,self.server_loss),'wb') as p:
                pickle.dump(self.server.state_dict(),p)
            os.mkdir(self.gru_hidden_dir + '{}/'.format(self.epoch))
            for i in range(self.num_sortingcenter):
                with open(self.gru_hidden_dir + '{}/'.format(self.epoch) + 'SortingCenter_{}.pkl'.format(i),'wb') as p:
                    pickle.dump(self.clients['SortingCenter'][i].skip_gru_hidden,p)
            for i in range(self.num_warehouse):
                with open(self.gru_hidden_dir + '{}/'.format(self.epoch) + 'Warehouse_{}.pkl'.format(i),'wb') as p:
                    pickle.dump(self.clients['Warehouse'][i].skip_gru_hidden,p)

                
    
    def Log_Test(self):
        pres_test = {
            "all":self.outall_test,
            "store":self.outwh_test,
            "pack":self.outpack_test,
            "sort":self.outsort_test
        }
        yts_test =  {
            "all":self.server_data['y_all_t_all'][312*self.N_train:],
            "store":self.server_data['y_wh_t_all'][312*self.N_train:],
            "pack":self.server_data['y_pack_t_all'][312*self.N_train:],
            "sort":self.server_data['y_sort_t_all'][312*self.N_train:]
        }
        self.server_metric_test = metric_all(pres_test,yts_test)
        self.server_loss_test = self.server_loss_test / self.N_test
        self.warehouse_loss_test = self.warehouse_loss_test / self.N_test / len(self.clients['Warehouse'])
        self.pack_loss_test = self.pack_loss_test / self.N_test / len(self.clients['Warehouse'])
        self.sortingcenter_loss_test = self.sortingcenter_loss_test / self.N_test / len(self.clients['SortingCenter'])
        
        print("TEST: Server Loss:{}".format(self.server_loss_test))
        print("TEST: Warehouse Client Loss:{}".format(self.warehouse_loss_test))
        print("TEST: Warehouse Client pack Loss:{}".format(self.pack_loss_test))
        print("TEST: SortingCenter Client Loss:{}".format(self.sortingcenter_loss_test))

        self.print_data(data="TEST: Server Loss:{}".format(self.server_loss_test),file=self.f_server_loss_test)
        self.print_data(data="TEST: Warehouse Client Loss:{}".format(self.warehouse_loss_test),file=self.f_warehouse_loss_test)
        self.print_data(data="TEST: Warehouse Client pack Loss:{}".format(self.pack_loss_test),file=self.f_pack_loss_test)
        self.print_data(data="TEST: SortingCenter Client Loss:{}".format(self.sortingcenter_loss_test),file=self.f_sortingcenter_loss_test)
        self.print_data(data="TEST: Server Mertics:"+str(self.server_metric_test),file=self.f_metric_test)
        
        
    def print_data(self,data,file):
        with open(file,'a+') as f:
            print(data,file=f)