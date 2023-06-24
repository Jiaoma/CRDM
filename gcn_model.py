from collections import OrderedDict
import torch
from torch._C import InterfaceType
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import numpy as np

from backbone import *
from utils import *
# from roi_align.roi_align import RoIAlign      # RoIAlign module
# from roi_align.roi_align import CropAndResize  # crop_and_resize module
from torchvision.ops import roi_align

from S3D_G import Pose3DEmbNet,TraceEmbNet,PoseTraceNet

# from torch_geometric_temporal.nn.recurrent import GConvGRU

class GCN_NN_Module(nn.Module):
    def __init__(self,cfg,number_app,number_mt):
        super(GCN_NN_Module, self).__init__()
        self.cfg=cfg
        number_app=number_app+number_mt
        self.number_app=number_app
        # self.number_mt=number_mt

        num_app=cfg.num_features_relation_app+cfg.num_features_relation_mt
        # num_mt=cfg.num_features_relation_mt
        self.fc_app=nn.Sequential(nn.Linear(number_app*2,number_app*2),nn.Linear(number_app*2,num_app))
        # self.fc_mt=nn.Sequential(nn.Linear(number_mt*2,number_mt*2),nn.Linear(number_mt*2,num_mt))
        self.nl_app= nn.LayerNorm([num_app])
        # self.nl_mt = nn.LayerNorm([num_mt])
        self.gru_app=nn.GRU(num_app,num_app,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) #
        # self.gru_mt=nn.GRU(num_mt,num_mt,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) #
        self.fc_final=nn.Sequential(nn.Linear(2*(num_app),2*(num_app)),nn.Linear(2*(num_app),cfg.num_features_relation))
        self.nl_final = nn.LayerNorm([cfg.num_features_relation])

    def forward(self, app_features,mt_features):
        B, T, N, _ = app_features.shape
        device=app_features.device
        # app_features=app_features[:,:,:,None,:].expand((B,T,N,N,self.number_app))
        # mt_features=mt_features[:,:,:,None,:].expand((B,T,N,N,self.number_mt))
        all_features=torch.cat([app_features,mt_features],dim=-1)
        all_features=all_features[:,:,:,None,:].expand((B,T,N,N,self.number_app))
        app_edge_feature= torch.relu(self.nl_app(self.fc_app(torch.cat([all_features,all_features.transpose(2,3)],dim=-1)))) # B,T,N,N,num_features_relation_app
        # mt_edge_feature= torch.relu(self.nl_mt(self.fc_mt(torch.cat([mt_features,mt_features.transpose(2,3)],dim=-1)))) # B,T,N,N,num_features_relation_mt
        
        app_edge_feature=app_edge_feature.transpose(0,1).reshape(T,B*N*N,self.cfg.num_features_relation_app+self.cfg.num_features_relation_mt)
        # mt_edge_feature=mt_edge_feature.transpose(0,1).reshape(T,B*N*N,self.cfg.num_features_relation_mt)
        # app_edge_feature=torch.cat([app_edge_feature,dis],dim=-1)
        
        h1 = torch.empty((2*2, B*N*N, self.cfg.num_features_relation_app+self.cfg.num_features_relation_mt), device=device)
        torch.nn.init.xavier_normal_(h1)

        self.gru_app.flatten_parameters()
        gru_outputs_app,_=self.gru_app(app_edge_feature,h1)

        gru_outputs_app=gru_outputs_app.reshape(T, B,N,N, 2*(self.cfg.num_features_relation_app+self.cfg.num_features_relation_mt)).transpose(0,1)
        
        return torch.relu(self.nl_final(self.fc_final(gru_outputs_app)))
    
class GCN_ARG_Module(nn.Module):
    def __init__(self, cfg, input_feature):
        super(GCN_ARG_Module, self).__init__()

        self.cfg = cfg

        NFR = cfg.num_features_relation

        NG = cfg.num_graph
        N = cfg.num_boxes
        T = cfg.num_frames

        NFG = cfg.num_features_gcn
        NFG_ONE = NFG

        self.fc_rn_theta_list = torch.nn.ModuleList([nn.Linear(input_feature, NFR) for i in range(NG)])
        self.fc_rn_phi_list = torch.nn.ModuleList([nn.Linear(input_feature, NFR) for i in range(NG)])

        self.fc_gcn_list = torch.nn.ModuleList([nn.Linear(input_feature, NFG_ONE, bias=False) for i in range(NG)])

        self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([T, N, NFG_ONE]) for i in range(NG)])
        
        self.gru_app=nn.GRU(NFG_ONE,NFG_ONE,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) #
        # self.gru_mt=nn.GRU(num_mt,num_mt,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) #
        self.fc_final=nn.Sequential(nn.Linear(2*(NFG_ONE),2*(NFG_ONE)),nn.Linear(2*(NFG_ONE),cfg.num_features_relation))
        self.nl_final = nn.LayerNorm([cfg.num_features_relation])
        self.gru_edge=nn.GRU(N*N,N*N,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) #

    def forward(self, graph_boxes_features):
        """
        graph_boxes_features  [B,T,N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        B, T, N, _ = graph_boxes_features.shape
        NFR = self.cfg.num_features_relation
        NG = self.cfg.num_graph
        NFG_ONE = self.cfg.num_features_gcn

        OH, OW = self.cfg.out_size
        device=graph_boxes_features.device


        relation_graph = None
        graph_boxes_features_list = []
        relation_graph_list=[]
        for i in range(NG):
            graph_boxes_features_theta = self.fc_rn_theta_list[i](graph_boxes_features)  # B,T,N,NFR
            graph_boxes_features_phi = self.fc_rn_phi_list[i](graph_boxes_features)  # B,T,N,NFR

            # graph_boxes_features_theta=self.nl_rn_theta_list[i](graph_boxes_features_theta)
            # graph_boxes_features_phi=self.nl_rn_phi_list[i](graph_boxes_features_phi)

            similarity_relation_graph = torch.matmul(graph_boxes_features_theta,
                                                     graph_boxes_features_phi.transpose(2, 3))  # B,T,N,N

            similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR)

            similarity_relation_graph = similarity_relation_graph.reshape(-1, 1)  # B*T*N*N, 1

            # Build relation graph
            relation_graph = similarity_relation_graph

            relation_graph = relation_graph.reshape(B,T, N, N)

            relation_graph = torch.softmax(relation_graph, dim=3)
            if True in torch.isnan(relation_graph):
                print('debug')
            relation_graph_list.append(relation_graph)

            # Graph convolution
            one_graph_boxes_features = self.fc_gcn_list[i](
                torch.matmul(relation_graph, graph_boxes_features))  # B,T, N, NFG_ONE
            one_graph_boxes_features = self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features = F.relu(one_graph_boxes_features, inplace=True)

            graph_boxes_features_list.append(one_graph_boxes_features)

        graph_boxes_features = torch.mean(torch.stack(graph_boxes_features_list), dim=0)  # B,T, N, NFG
        relation_graph_list=torch.mean(torch.stack(relation_graph_list),dim=0).transpose(0,1).reshape(T,B,N*N) # T,B* N* N,1
        if True in torch.isnan(relation_graph_list):
                print('debug')
        h1 = torch.empty((2*2, B*N, NFG_ONE), device=device)
        torch.nn.init.xavier_normal_(h1)
        graph_boxes_features=graph_boxes_features.transpose(0,1).reshape(T,B*N,NFG_ONE)
        self.gru_app.flatten_parameters()
        gru_outputs_app,_=self.gru_app(graph_boxes_features,h1)

        gru_outputs_app=gru_outputs_app.reshape(T, B,N, 2*(NFG_ONE)).transpose(0,1)
        
        h2 = torch.empty((2*2, B, N*N), device=device)
        torch.nn.init.xavier_normal_(h2)

        self.gru_edge.flatten_parameters()
        relation_graphs,_=self.gru_edge(relation_graph_list,h2)
        if True in torch.isnan(relation_graphs):
                print('debug')

        relation_graphs=relation_graphs.reshape(T, B,2,N*N).transpose(0,1).mean(-2) # B T N N
        if True in torch.isnan(relation_graphs):
                print('debug')
        if True in torch.isnan(gru_outputs_app):
                print('debug')
        return torch.relu(self.nl_final(self.fc_final(gru_outputs_app))), relation_graphs.reshape(B,T,N,N)
    
    
class GCN_NN_Module_wolong(nn.Module):
    def __init__(self,cfg,number_app,number_mt):
        super(GCN_NN_Module_wolong, self).__init__()
        self.cfg=cfg
        number_app=number_app+number_mt
        self.number_app=number_app
        # self.number_mt=number_mt

        num_app=cfg.num_features_relation_app+cfg.num_features_relation_mt
        # num_mt=cfg.num_features_relation_mt
        self.fc_app=nn.Sequential(nn.Linear(number_app*2,number_app*2),nn.Linear(number_app*2,num_app))
        # self.fc_mt=nn.Sequential(nn.Linear(number_mt*2,number_mt*2),nn.Linear(number_mt*2,num_mt))
        self.nl_app= nn.LayerNorm([num_app])
        # self.nl_mt = nn.LayerNorm([num_mt])
        # self.gru_app=nn.GRU(num_app,num_app,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) #
        # self.gru_mt=nn.GRU(num_mt,num_mt,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) #
        self.fc_final=nn.Sequential(nn.Linear(self.cfg.num_features_relation_app+self.cfg.num_features_relation_mt,self.cfg.num_features_relation_app+self.cfg.num_features_relation_mt),nn.Linear(self.cfg.num_features_relation_app+self.cfg.num_features_relation_mt,cfg.num_features_relation))
        self.nl_final = nn.LayerNorm([cfg.num_features_relation])

    def forward(self, app_features,mt_features):
        B, T, N, _ = app_features.shape
        device=app_features.device
        # app_features=app_features[:,:,:,None,:].expand((B,T,N,N,self.number_app))
        # mt_features=mt_features[:,:,:,None,:].expand((B,T,N,N,self.number_mt))
        all_features=torch.cat([app_features,mt_features],dim=-1)
        all_features=all_features[:,:,:,None,:].expand((B,T,N,N,self.number_app))
        app_edge_feature= torch.relu(self.nl_app(self.fc_app(torch.cat([all_features,all_features.transpose(2,3)],dim=-1)))) # B,T,N,N,num_features_relation_app
        # mt_edge_feature= torch.relu(self.nl_mt(self.fc_mt(torch.cat([mt_features,mt_features.transpose(2,3)],dim=-1)))) # B,T,N,N,num_features_relation_mt
        
        # app_edge_feature=app_edge_feature.transpose(0,1).reshape(T,B*N*N,self.cfg.num_features_relation_app+self.cfg.num_features_relation_mt)
        # # mt_edge_feature=mt_edge_feature.transpose(0,1).reshape(T,B*N*N,self.cfg.num_features_relation_mt)
        
        # h1 = torch.empty((2*2, B*N*N, self.cfg.num_features_relation_app+self.cfg.num_features_relation_mt), device=device)
        # torch.nn.init.xavier_normal_(h1)

        # self.gru_app.flatten_parameters()
        # gru_outputs_app,_=self.gru_app(app_edge_feature,h1)

        # gru_outputs_app=gru_outputs_app.reshape(T, B,N,N, 2*(self.cfg.num_features_relation_app+self.cfg.num_features_relation_mt)).transpose(0,1)
        
        # h2 = torch.empty((2*2, B*N*N, self.cfg.num_features_relation_mt), device=device)
        # torch.nn.init.xavier_normal_(h2)

        # self.gru_mt.flatten_parameters()
        # gru_outputs_mt,_=self.gru_mt(mt_edge_feature,h2)

        # gru_outputs_mt=gru_outputs_mt.reshape(T, B,N,N, 2*self.cfg.num_features_relation_mt).transpose(0,1)
        
        return torch.relu(self.nl_final(self.fc_final(app_edge_feature)))

class GCN_NN_Module_wotraj(nn.Module):
    def __init__(self,cfg,number_app,number_mt):
        super(GCN_NN_Module_wotraj, self).__init__()
        self.cfg=cfg
        self.number_app=number_app
        self.number_mt=number_mt
        num_app=cfg.num_features_relation_app
        num_mt=cfg.num_features_relation_mt
        self.fc_app=nn.Sequential(nn.Linear(number_app*2,number_app*2),nn.Linear(number_app*2,num_app))
        # self.fc_mt=nn.Sequential(nn.Linear(number_mt*2,number_mt*2),nn.Linear(number_mt*2,num_mt))
        self.nl_app= nn.LayerNorm([num_app])
        # self.nl_mt = nn.LayerNorm([num_mt])
        self.gru_app=nn.GRU(num_app,num_app,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) #
        # self.gru_mt=nn.GRU(num_mt,num_mt,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) #
        self.fc_final=nn.Sequential(nn.Linear(2*(num_app),2*(num_app)),nn.Linear(2*(num_app),cfg.num_features_relation))
        self.nl_final = nn.LayerNorm([cfg.num_features_relation])

    def forward(self, app_features):
        B, T, N, _ = app_features.shape
        device=app_features.device
        app_features=app_features[:,:,:,None,:].expand((B,T,N,N,self.number_app))
        app_edge_feature= torch.relu(self.nl_app(self.fc_app(torch.cat([app_features,app_features.transpose(2,3)],dim=-1)))) # B,T,N,N,num_features_relation_app
        
        app_edge_feature=app_edge_feature.transpose(0,1).reshape(T,B*N*N,self.cfg.num_features_relation_app)
        
        h1 = torch.empty((2*2, B*N*N, self.cfg.num_features_relation_app), device=device)
        torch.nn.init.xavier_normal_(h1)

        self.gru_app.flatten_parameters()
        gru_outputs_app,_=self.gru_app(app_edge_feature,h1)

        gru_outputs_app=gru_outputs_app.reshape(T, B,N,N, 2*(self.cfg.num_features_relation_app)).transpose(0,1)
        
        return torch.relu(self.nl_final(self.fc_final(gru_outputs_app)))
    
class GCN_NN_Module_woapp(nn.Module):
    def __init__(self,cfg,number_app,number_mt):
        super(GCN_NN_Module_woapp, self).__init__()
        self.cfg=cfg
        self.number_app=number_app
        self.number_mt=number_mt
        num_app=cfg.num_features_relation_app
        num_mt=cfg.num_features_relation_mt
        # self.fc_app=nn.Sequential(nn.Linear(number_app*2,number_app*2),nn.Linear(number_app*2,num_app+num_mt))
        self.fc_mt=nn.Sequential(nn.Linear(number_mt*2,number_mt*2),nn.Linear(number_mt*2,num_mt))
        # self.nl_app= nn.LayerNorm([num_app+num_mt])
        self.nl_mt = nn.LayerNorm([num_mt])
        # self.gru_app=nn.GRU(num_app+num_mt,num_app+num_mt,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) #
        self.gru_mt=nn.GRU(num_mt,num_mt,num_layers=2,bidirectional=True,dropout=self.cfg.train_dropout_prob) 
        self.fc_final=nn.Sequential(nn.Linear(2*(num_mt),2*(num_mt)),nn.Linear(2*(num_mt),cfg.num_features_relation))
        self.nl_final = nn.LayerNorm([cfg.num_features_relation])

    def forward(self, mt_features):
        B, T, N, _ = mt_features.shape
        device=mt_features.device
        mt_features=mt_features[:,:,:,None,:].expand((B,T,N,N,self.number_mt))
        mt_edge_feature= torch.relu(self.nl_mt(self.fc_mt(torch.cat([mt_features,mt_features.transpose(2,3)],dim=-1)))) # B,T,N,N,num_features_relation_app
        
        mt_edge_feature=mt_edge_feature.transpose(0,1).reshape(T,B*N*N,self.cfg.num_features_relation_mt)
        
        h1 = torch.empty((2*2, B*N*N, self.cfg.num_features_relation_mt), device=device)
        torch.nn.init.xavier_normal_(h1)

        self.gru_mt.flatten_parameters()
        gru_outputs_mt,_=self.gru_mt(mt_edge_feature,h1)

        gru_outputs_mt=gru_outputs_mt.reshape(T, B,N,N, 2*(self.cfg.num_features_relation_mt)).transpose(0,1)
        
        return torch.relu(self.nl_final(self.fc_final(gru_outputs_mt)))

from S3D_G import BasicConv3d

class OneGroupV5Net_image(nn.Module):
    """
    main module of GCN for the collective dataset
    """

    def __init__(self, cfg):
        super(OneGroupV5Net_image, self).__init__()
        ################ Parameters #####################
        self.cfg = cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
        MAX_N = self.cfg.num_boxes
        self.R2 = nn.Parameter(torch.tensor([_i for _i in range(1,MAX_N+1)],dtype=torch.float32), requires_grad=False)[None,:] # 1,2,3,4,...
        ################ Parameters #####################

        ################ Modules ########################
        self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        assert self.cfg.continuous_frames>=3 and self.cfg.continuous_frames%2==1
        conv3d=[]
        for i in range((self.cfg.continuous_frames-1)//2):
            conv3d.append(BasicConv3d(D,D, kernel_size=(3,1,1),stride=1))
        self.conv3d_app=nn.Sequential(*conv3d)
        self.fc_app=nn.Linear(D*K*K,self.cfg.num_features_pose)
        self.nl_emb_1 = nn.LayerNorm([self.cfg.num_features_pose])
        # graph update node
        self.graph=GCN_NN_Module(cfg,self.cfg.num_features_pose,2+2)
        # output action
        self.fc_actions = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_actions)
        )
        # use actions, plus relation, output which interaction, output who are the interactors.
        self.fc_actors = torch.nn.Sequential(
            nn.Linear(number_feature_relation,4*number_feature_relation),
            nn.Linear(4*number_feature_relation, 1)
        )
        self.fc_activities = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_activities)
        )
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        ################ Modules ########################
        
        
        ################ Init ###########################
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
        ################ Init ###########################
    def loadmodel(self, filepath):
        state = torch.load(filepath)
        backbone = OrderedDict()
        for key in state['backbone_state_dict'].keys():
            if 'Mixed_6' in key:
                continue
            else:
                backbone[key] = state['backbone_state_dict'][key]
        self.backbone.load_state_dict(backbone)
        print('Load model states from: ', filepath)

    def loadmodel_normal(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict'][key]
        self.load_state_dict(process_dict)
        print('Load all parameter from: ', filepath)
    @autocast()
    def forward(self, batch_data, ablation=False):
        with autocast():
            if ablation:
                images_in, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data  # B,T,MAX_N
            else:
                images_in, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data
            # B,T,F,C,H,W   B,T,MAX_N,4,F |  B,T     | B,T,MAX_N,2,F
            '''
            If in training:
            B,3T,MAX_N,D*F*K*K   B,3T,MAX_N,2,F |  B,3T      | B,3T,MAX_N,2,F
            3T: origin, positive, negative
            output of shape (seq_len, batch, num_directions * hidden_size)

            h_n of shape (num_layers * num_directions, batch, hidden_size)
            h_n.view(num_layers, num_directions, batch, hidden_size).
            '''

            # read config parameters
            B = boxes_in.shape[0]
            T = boxes_in.shape[1]
            H, W = self.cfg.image_size
            OH, OW = self.cfg.out_size
            D = self.cfg.emb_features
            K = self.cfg.crop_size[0]
            F = self.cfg.continuous_frames
            device = boxes_in.device
            if self.R2.device!=device:
                self.R2=self.R2.to(device)

            MAX_N = self.cfg.num_boxes
            number_feature_boxes = self.cfg.num_features_boxes
            num_features_gcn = self.cfg.num_features_gcn
            number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
            Agg_num=3*num_features_gcn+4 

            if self.training:
                B = B * 3
                T = T // 3
                images_in.resize_((B, T) + images_in.shape[2:])
                boxes_in.resize_((B, T) + boxes_in.shape[2:])
                bboxes_num_in.resize_((B, T))
                boxes_dhdw.resize_((B, T) + boxes_dhdw.shape[2:])
                if ablation:
                    actor_label.resize_((B, T,MAX_N,MAX_N))

            # Reshape the input data
            images_in_flat = torch.reshape(images_in, (B * T*F, 3, H, W))  # B*T, 3, H, W
            boxes_in_flat = boxes_in.permute(0,1,4,2,3).reshape(B*T*F*MAX_N,4)
            boxes_in_flat=torch.split(boxes_in_flat,MAX_N,dim=0)

            # Use backbone to extract features of images_in
            # Pre-precess first
            images_in_flat = prep_images(images_in_flat)
            outputs = self.backbone(images_in_flat)

            # Build multiscale features
            features_multiscale = []
            for features in outputs:
                if features.shape[2:4] != torch.Size([OH, OW]):
                    features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
                features_multiscale.append(features)

            features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T*F, D, OH, OW

            with torch.no_grad():
                pose_features = roi_align(features_multiscale,
                                                    boxes_in_flat,
                                                    self.cfg.crop_size)  # B*T*MAX_N, D, K, K,
                
                pose_features = pose_features.reshape(B,T,F, MAX_N, D,K,K).permute(0,1,3,4,2,5,6)  # B,T,MAX_N, D,F,K,K
                
                boxes_center = (boxes_in[:, :, :, :2, 0] + boxes_in[:, :, :, 2:, 0]) / 2

                # boxes_center=boxes_center.transpose(3,4)#B,T,MAX_N,2
                boxes_dhdw=boxes_dhdw[...,0]#B,T,MAX_N,2
            
                boxes_exist=boxes_center.sum(dim=3)
                boxes_exist=boxes_exist>0 # B,T,N
                boxes_exist_int=boxes_exist.float()
                # Below is the old version, it's totally wrong
                boxes_exist_matrix=boxes_exist[:,:,:,None,None]*(boxes_exist[:,:,None,:,None]) # B,T,MAX_N,MAX_N,1
                boxes_exist_matrix_line=boxes_exist[:,:,:,None,None].expand(B,T,MAX_N,MAX_N,1) # B,T,MAX_N,MAX_N,1
                conflicts=torch.matmul(boxes_exist_int.transpose(1,2),boxes_exist_int)
                conflicts=conflicts==0 # B,N,N     main dim is the second N
                # bboxes_num_in=torch.zeros((B,),dtype=torch.int)
                temp=boxes_exist.sum(dim=1) #B,N
                temp=(temp>0).float()
                
                temp=temp*self.R2
                bboxes_num_in=(torch.argmax(temp,dim=1)+1).tolist() # B
            all_features=torch.cat([pose_features.reshape(B,T,MAX_N,D*F*K*K),boxes_center,boxes_dhdw],dim=3)
            
            # Calculate mean node features
            mean_node_features=torch.sum(all_features*boxes_exist[...,None],dim=1,keepdim=True)/torch.sum(boxes_exist[...,None].float(),dim=1,keepdim=True) # B,MAX_N,NFG
            # f=lambda x,y: torch.abs(x-y).sum(dim=-1)
            mask=~boxes_exist
            select=torch.nonzero(mask).tolist()
            for (b,t,box) in select:
                if box>=bboxes_num_in[b]:
                    continue
                if mask[b,t,box]:
                    other_node_index=conflicts[b,:,box]
                    have_other=other_node_index.sum()
                    if have_other:
                        # local_node_graph_features=node_graph_features[b,t]
                        other_node_features=all_features[b,t,other_node_index] # X, NFG 
                        min_distant_index=torch.argmin(
                            torch.abs(mean_node_features[b,:,box,:]-other_node_features).sum(dim=-1)) # 1
                        all_features[b,t,box].add_(other_node_features[min_distant_index]) # Copy here.
                    else:
                        all_features[b,t,box].add_(mean_node_features[b,0,box])
            
            pose_features,boxes_center,boxes_dhdw=all_features[...,:D*F*K*K],all_features[...,D*F*K*K:D*F*K*K+2],all_features[...,D*F*K*K+2:]
            
            del all_features
            del conflicts
            del mask
            del temp
            del mean_node_features
            del select
            pose_features = pose_features.reshape(B*T*MAX_N, D, F, K, K)
            pose_features=self.conv3d_app(pose_features).reshape(B*T*MAX_N, D*K*K) # aggregate F
            app_features=self.fc_app(pose_features).reshape(B,T,MAX_N,self.cfg.num_features_pose)
            app_features=self.nl_emb_1(app_features)
            app_features=torch.relu(app_features)
            
            mt_features=torch.cat([boxes_center,boxes_dhdw],dim=3) # B,T,MAX_N,2+2
            edge_feature_matrix=self.graph(app_features,mt_features) # B,T,MAX_N,MAX_N,num_features_relation
            
            # # Max pooling.
            # edge_feature_matrix=torch.max(edge_feature_matrix,dim=1,keepdim=True)[0].expand(B,T,MAX_N,MAX_N,self.cfg.num_features_relation)
            

            actor_score=self.fc_actors(edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N)
            actor_score=torch.softmax(actor_score,dim=-1).reshape(B,T,MAX_N,MAX_N) # B,T,N,N 
            # Below is the old version, it's not correct for the person-disappear cases.
            # actor_score=(actor_score+actor_score.transpose(-1,-2))/2
            
            boxes_states_pooled=torch.max((edge_feature_matrix*boxes_exist_matrix_line).reshape(B,T,MAX_N*MAX_N,self.cfg.num_features_relation),dim=2)[0]
            acty_score=self.fc_activities(boxes_states_pooled)
            
            node_states_pooled=torch.max(edge_feature_matrix,dim=3)[0]
            action_score= self.fc_actions(node_states_pooled)
            
            if self.training:
                return action_score,acty_score, actor_score, boxes_states_pooled.reshape(B,T,-1)  # Here, b is 3 times as the original b.
            else:
                return action_score,acty_score, actor_score  # Here, b is 3 times as the original b.
            
class OneGroupV5Net_image_actor(nn.Module):
    """
    main module of GCN for the collective dataset
    """

    def __init__(self, cfg):
        super(OneGroupV5Net_image_actor, self).__init__()
        ################ Parameters #####################
        self.cfg = cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
        MAX_N = self.cfg.num_boxes
        self.R2 = nn.Parameter(torch.tensor([_i for _i in range(1,MAX_N+1)],dtype=torch.float32), requires_grad=False)[None,:] # 1,2,3,4,...
        ################ Parameters #####################

        ################ Modules ########################
        self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        assert self.cfg.continuous_frames>=3 and self.cfg.continuous_frames%2==1
        conv3d=[]
        for i in range((self.cfg.continuous_frames-1)//2):
            conv3d.append(BasicConv3d(D,D, kernel_size=(3,1,1),stride=1))
        self.conv3d_app=nn.Sequential(*conv3d)
        self.fc_app=nn.Linear(D*K*K,self.cfg.num_features_pose)
        self.nl_emb_1 = nn.LayerNorm([self.cfg.num_features_pose])
        # graph update node
        self.graph=GCN_NN_Module(cfg,self.cfg.num_features_pose,2+2)
        # output action
        self.fc_actions = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_actions)
        )
        # use actions, plus relation, output which interaction, output who are the interactors.
        self.fc_actors = torch.nn.Sequential(
            nn.Linear(number_feature_relation,4*number_feature_relation),
            nn.Linear(4*number_feature_relation, 1)
        )
        self.fc_activities = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_activities)
        )
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        ################ Modules ########################
        
        
        ################ Init ###########################
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
        ################ Init ###########################
    def loadmodel(self, filepath):
        state = torch.load(filepath)
        backbone = OrderedDict()
        for key in state['backbone_state_dict'].keys():
            if 'Mixed_6' in key:
                continue
            else:
                backbone[key] = state['backbone_state_dict'][key]
        self.backbone.load_state_dict(backbone)
        print('Load model states from: ', filepath)

    def loadmodel_normal(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict'][key]
        self.load_state_dict(process_dict)
        print('Load all parameter from: ', filepath)
    @autocast()
    def forward(self, batch_data, ablation=False,vis=None):
        with autocast():
            if ablation:
                images_in, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data  # B,T,MAX_N
            else:
                images_in, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data
            # B,T,F,C,H,W   B,T,MAX_N,4,F |  B,T     | B,T,MAX_N,2,F
            '''
            If in training:
            B,3T,MAX_N,D*F*K*K   B,3T,MAX_N,2,F |  B,3T      | B,3T,MAX_N,2,F
            3T: origin, positive, negative
            output of shape (seq_len, batch, num_directions * hidden_size)

            h_n of shape (num_layers * num_directions, batch, hidden_size)
            h_n.view(num_layers, num_directions, batch, hidden_size).
            '''

            # read config parameters
            B = boxes_in.shape[0]
            T = boxes_in.shape[1]
            H, W = self.cfg.image_size
            OH, OW = self.cfg.out_size
            D = self.cfg.emb_features
            K = self.cfg.crop_size[0]
            F = self.cfg.continuous_frames
            device = boxes_in.device
            if self.R2.device!=device:
                self.R2=self.R2.to(device)

            MAX_N = self.cfg.num_boxes
            number_feature_boxes = self.cfg.num_features_boxes
            num_features_gcn = self.cfg.num_features_gcn
            number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
            Agg_num=3*num_features_gcn+4 

            if self.training:
                B = B * 3
                T = T // 3
                images_in.resize_((B, T) + images_in.shape[2:])
                boxes_in.resize_((B, T) + boxes_in.shape[2:])
                bboxes_num_in.resize_((B, T))
                boxes_dhdw.resize_((B, T) + boxes_dhdw.shape[2:])
                if ablation:
                    actor_label.resize_((B, T,MAX_N,MAX_N))

            # Reshape the input data
            images_in_flat = torch.reshape(images_in, (B * T*F, 3, H, W))  # B*T, 3, H, W
            boxes_in_flat = boxes_in.permute(0,1,4,2,3).reshape(B*T*F*MAX_N,4)
            boxes_in_flat=torch.split(boxes_in_flat,MAX_N,dim=0)

            # Use backbone to extract features of images_in
            # Pre-precess first
            images_in_flat = prep_images(images_in_flat)
            outputs = self.backbone(images_in_flat)

            # Build multiscale features
            features_multiscale = []
            for features in outputs:
                if features.shape[2:4] != torch.Size([OH, OW]):
                    features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
                features_multiscale.append(features)

            features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T*F, D, OH, OW

            with torch.no_grad():
                pose_features = roi_align(features_multiscale,
                                                    boxes_in_flat,
                                                    self.cfg.crop_size)  # B*T*MAX_N, D, K, K,
                
                pose_features = pose_features.reshape(B,T,F, MAX_N, D,K,K).permute(0,1,3,4,2,5,6)  # B,T,MAX_N, D,F,K,K
                
                boxes_center = (boxes_in[:, :, :, :2, 0] + boxes_in[:, :, :, 2:, 0]) / 2

                # boxes_center=boxes_center.transpose(3,4)#B,T,MAX_N,2
                boxes_dhdw=boxes_dhdw[...,0]#B,T,MAX_N,2
            
                boxes_exist=boxes_center.sum(dim=3)
                boxes_exist=boxes_exist>0 # B,T,N
                boxes_exist_int=boxes_exist.float()
                # Below is the old version, it's totally wrong
                boxes_exist_matrix=boxes_exist[:,:,:,None,None]*(boxes_exist[:,:,None,:,None]) # B,T,MAX_N,MAX_N,1
                boxes_exist_matrix_line=boxes_exist[:,:,:,None,None].expand(B,T,MAX_N,MAX_N,1) # B,T,MAX_N,MAX_N,1
                conflicts=torch.matmul(boxes_exist_int.transpose(1,2),boxes_exist_int)
                conflicts=conflicts==0 # B,N,N     main dim is the second N
                # bboxes_num_in=torch.zeros((B,),dtype=torch.int)
                temp=boxes_exist.sum(dim=1) #B,N
                temp=(temp>0).float()
                
                temp=temp*self.R2
                bboxes_num_in=(torch.argmax(temp,dim=1)+1).tolist() # B
            all_features=torch.cat([pose_features.reshape(B,T,MAX_N,D*F*K*K),boxes_center,boxes_dhdw],dim=3)
            all_features=all_features*boxes_exist[...,None]
            
            # Calculate mean node features
            mean_node_features=torch.sum(all_features,dim=1,keepdim=True)/torch.sum(boxes_exist[...,None].float(),dim=1,keepdim=True) # B,MAX_N,NFG
            mean_node_features[mean_node_features!=mean_node_features]=0
            # f=lambda x,y: torch.abs(x-y).sum(dim=-1)
            mask=~boxes_exist
            select=torch.nonzero(mask).tolist()
            for (b,t,box) in select:
                if box>=bboxes_num_in[b]:
                    continue
                if mask[b,t,box]:
                    other_node_index=conflicts[b,:,box]
                    exist_at_least_once=boxes_exist[b,:,box].sum()
                    have_other=other_node_index.sum()
                    if have_other and exist_at_least_once:
                        # local_node_graph_features=node_graph_features[b,t]
                        other_node_features=all_features[b,t,other_node_index] # X, NFG 
                        min_distant_index=torch.argmin(
                            torch.abs(mean_node_features[b,:,box,:]-other_node_features).sum(dim=-1)) # 1
                        all_features[b,t,box].add_(other_node_features[min_distant_index]) # Copy here.
                    else:
                        all_features[b,t,box].add_(mean_node_features[b,0,box])
            
            pose_features,boxes_center,boxes_dhdw=all_features[...,:D*F*K*K],all_features[...,D*F*K*K:D*F*K*K+2],all_features[...,D*F*K*K+2:]
            
            del all_features
            del conflicts
            del mask
            del temp
            del mean_node_features
            del select
            pose_features = pose_features.reshape(B*T*MAX_N, D, F, K, K)
            pose_features=self.conv3d_app(pose_features).reshape(B*T*MAX_N, D*K*K) # aggregate F
            app_features=self.fc_app(pose_features).reshape(B,T,MAX_N,self.cfg.num_features_pose)
            app_features=self.nl_emb_1(app_features)
            app_features=torch.relu(app_features)
            
            mt_features=torch.cat([boxes_center,boxes_dhdw],dim=3) # B,T,MAX_N,2+2
            edge_feature_matrix=self.graph(app_features,mt_features) # B,T,MAX_N,MAX_N,num_features_relation
            
            # # Max pooling.
            # edge_feature_matrix=torch.max(edge_feature_matrix,dim=1,keepdim=True)[0].expand(B,T,MAX_N,MAX_N,self.cfg.num_features_relation)
            
            actor_score=self.fc_actors(edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N)
            actor_score=torch.softmax(actor_score,dim=-1).reshape(B,T,MAX_N,MAX_N) # B,T,N,N 
            # Below is the old version, it's not correct for the person-disappear cases.
            # actor_score=(actor_score+actor_score.transpose(-1,-2))/2
            
            boxes_states_pooled=torch.sum((edge_feature_matrix*actor_score[...,None]).reshape(B,T,MAX_N*MAX_N,self.cfg.num_features_relation),dim=2)
            acty_score=self.fc_activities(boxes_states_pooled)
            
            node_states_pooled=torch.sum(edge_feature_matrix*actor_score[...,None],dim=3)+torch.diagonal(edge_feature_matrix*boxes_exist_matrix,dim1=-3,dim2=-2).transpose(-1,-2)
            action_score= self.fc_actions(node_states_pooled)
            
            if vis is not None:
                vis.plotActorMatrix(actor_label[0],actor_score[0])
            
            if self.training:
                return action_score,acty_score, actor_score, boxes_states_pooled.reshape(B,T,-1)  # Here, b is 3 times as the original b.
            else:
                return action_score,acty_score, actor_score  # Here, b is 3 times as the original b.

class OneGroupV5Net(nn.Module):
    """
    main module of GCN for the collective dataset
    """

    def __init__(self, cfg):
        super(OneGroupV5Net, self).__init__()
        ################ Parameters #####################
        self.cfg = cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
        MAX_N = self.cfg.num_boxes
        self.R2 = nn.Parameter(torch.tensor([_i for _i in range(1,MAX_N+1)],dtype=torch.float32), requires_grad=False)[None,:] # 1,2,3,4,...
        ################ Parameters #####################

        ################ Modules ########################
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        # if not self.cfg.train_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False
        assert self.cfg.continuous_frames>=3 and self.cfg.continuous_frames%2==1
        conv3d=[]
        for i in range((self.cfg.continuous_frames-1)//2):
            conv3d.append(BasicConv3d(D,D, kernel_size=(3,1,1),stride=1))
        self.conv3d_app=nn.Sequential(*conv3d)
        self.fc_app=nn.Linear(D*K*K,self.cfg.num_features_pose)
        self.nl_emb_1 = nn.LayerNorm([self.cfg.num_features_pose])
        self.gru_mt=nn.GRU(2+2,self.cfg.num_features_mt,num_layers=1,bidirectional=False)
        # graph update node
        self.graph=GCN_NN_Module(cfg,self.cfg.num_features_pose,self.cfg.num_features_mt)
        # output action
        self.fc_actions = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_actions)
        )
        #
        # use actions, plus relation, output which interaction, output who are the interactors.
        self.fc_actors = torch.nn.Sequential(
            nn.Linear(number_feature_relation,4*number_feature_relation),
            nn.Linear(4*number_feature_relation, 1)
        )
        self.fc_activities = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_activities)
        )
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        ################ Modules ########################
        
        
        ################ Init ###########################
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
        ################ Init ###########################
    def loadmodel(self, filepath):
        state = torch.load(filepath)
        backbone = OrderedDict()
        for key in state['backbone_state_dict'].keys():
            if 'Mixed_6' in key:
                continue
            else:
                backbone[key] = state['backbone_state_dict'][key]
        self.backbone.load_state_dict(backbone)
        print('Load model states from: ', filepath)

    def loadmodel_normal(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict'][key]
        self.load_state_dict(process_dict)
        print('Load all parameter from: ', filepath)
    @autocast()
    def forward(self, batch_data, ablation=False):
        with autocast():
            if ablation:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data  # B,T,MAX_N
            else:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data
            # B,T,F,C,H,W   B,T,MAX_N,4,F |  B,T     | B,T,MAX_N,2,F
            '''
            If in training:
            B,3T,MAX_N,D*F*K*K   B,3T,MAX_N,2,F |  B,3T      | B,3T,MAX_N,2,F
            3T: origin, positive, negative
            output of shape (seq_len, batch, num_directions * hidden_size)

            h_n of shape (num_layers * num_directions, batch, hidden_size)
            h_n.view(num_layers, num_directions, batch, hidden_size).
            '''

            # read config parameters
            B = boxes_in.shape[0]
            T = boxes_in.shape[1]
            H, W = self.cfg.image_size
            OH, OW = self.cfg.out_size
            D = self.cfg.emb_features
            K = self.cfg.crop_size[0]
            F = self.cfg.continuous_frames
            device = boxes_in.device
            if self.R2.device!=device:
                self.R2=self.R2.to(device)

            MAX_N = self.cfg.num_boxes
            number_feature_boxes = self.cfg.num_features_boxes
            num_features_gcn = self.cfg.num_features_gcn
            number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
            Agg_num=3*num_features_gcn+4 

            if self.training:
                B = B * 3
                T = T // 3
                pose_features.resize_((B, T) + pose_features.shape[2:])
                boxes_in.resize_((B, T) + boxes_in.shape[2:])
                bboxes_num_in.resize_((B, T))
                boxes_dhdw.resize_((B, T) + boxes_dhdw.shape[2:])
                if ablation:
                    actor_label.resize_((B, T,MAX_N,MAX_N))


            with torch.no_grad():
                pose_features = pose_features.reshape(B,T,MAX_N, D,F,K,K)  # B,T,MAX_N, D,F,K,K
                
                boxes_center = (boxes_in[:, :, :, :2,:] + boxes_in[:, :, :, 2:, :]) / 2 # B,T,N,2,F

                # boxes_center=boxes_center.transpose(3,4)#B,T,MAX_N,2
                boxes_dhdw=boxes_dhdw[...]#B,T,MAX_N,2,F
            
                boxes_exist=boxes_center[...,0].sum(dim=3)
                boxes_exist=boxes_exist>0 # B,T,N
                boxes_exist_int=boxes_exist.float()
                # Below is the old version, it's totally wrong
                boxes_exist_matrix=boxes_exist[:,:,:,None,None]*(boxes_exist[:,:,None,:,None]) # B,T,MAX_N,MAX_N,1
                boxes_exist_matrix_line=boxes_exist[:,:,:,None,None].expand(B,T,MAX_N,MAX_N,1) # B,T,MAX_N,MAX_N,1
                conflicts=torch.matmul(boxes_exist_int.transpose(1,2),boxes_exist_int)
                conflicts=conflicts==0 # B,N,N     main dim is the second N
                # bboxes_num_in=torch.zeros((B,),dtype=torch.int)
                temp=boxes_exist.sum(dim=1) #B,N
                temp=(temp>0).float()
                
                temp_index=temp*self.R2
                bboxes_num_in=(torch.argmax(temp_index,dim=1)+1).tolist() # B
            all_features=torch.cat([pose_features.reshape(B,T,MAX_N,D*F*K*K),boxes_center.reshape(B,T,MAX_N,2*F),boxes_dhdw.reshape(B,T,MAX_N,2*F)],dim=3)
            
            # Calculate mean node features
            mean_node_features=torch.sum(all_features*boxes_exist[...,None],dim=1,keepdim=True)/(torch.sum(boxes_exist[...,None].float(),dim=1,keepdim=True)+1e-12) # B,MAX_N,NFG
            # f=lambda x,y: torch.abs(x-y).sum(dim=-1)
            mask=~boxes_exist*temp[:,None,:]
            select=torch.nonzero(mask).tolist()
            for (b,t,box) in select:
                if box>=bboxes_num_in[b]:
                    continue
                if mask[b,t,box]:
                    other_node_index=conflicts[b,:,box]&boxes_exist[b,t]
                    have_other=other_node_index.sum()
                    if have_other:
                        # local_node_graph_features=node_graph_features[b,t]
                        other_node_features=all_features[b,t,other_node_index] # X, NFG 
                        min_distant_index=torch.argmin(
                            torch.abs(mean_node_features[b,:,box,:]-other_node_features).sum(dim=-1)) # 1
                        all_features[b,t,box].add_(other_node_features[min_distant_index]) # Copy here.
                    else:
                        all_features[b,t,box].add_(mean_node_features[b,0,box])
            
            pose_features,boxes_center,boxes_dhdw=all_features[...,:D*F*K*K],all_features[...,D*F*K*K:D*F*K*K+2*F],all_features[...,D*F*K*K+2*F:]
            
            del all_features
            del conflicts
            del mask
            del temp
            del mean_node_features
            del select
            pose_features = pose_features.reshape(B*T*MAX_N, D, F, K, K)
            pose_features=self.conv3d_app(pose_features).reshape(B*T*MAX_N, D*K*K) # aggregate F
            app_features=self.fc_app(pose_features).reshape(B,T,MAX_N,self.cfg.num_features_pose)
            app_features=self.nl_emb_1(app_features)
            app_features=torch.relu(app_features)
            
            h1 = torch.empty((1, B*T*MAX_N, self.cfg.num_features_mt), device=device)
            torch.nn.init.xavier_normal_(h1)

            self.gru_mt.flatten_parameters()
            # dis=boxes_center.reshape(B,T,MAX_N,2,F)[...,0]
            # dis=((dis[:,:,:,None,:].expand(B,T,MAX_N,MAX_N,2)-dis[:,:,None,:,:].expand(B,T,MAX_N,MAX_N,2))**2).sum(dim=-1,keepdim=True).transpose(0,1).reshape(T,B*MAX_N*MAX_N,1)
            mt_features=torch.cat([boxes_center.reshape(B,T,MAX_N,2,F),boxes_dhdw.reshape(B,T,MAX_N,2,F)],dim=3).permute(4,0,1,2,3).reshape(F,B*T*MAX_N,4) # F,B,T,MAX_N,4
            _,mt_features=self.gru_mt(mt_features,h1)
            mt_features=mt_features.reshape(B,T,MAX_N,self.cfg.num_features_mt)
            edge_feature_matrix=self.graph(app_features,mt_features) # B,T,MAX_N,MAX_N,num_features_relation
            
            # # Max pooling.
            # edge_feature_matrix=torch.max(edge_feature_matrix,dim=1,keepdim=True)[0].expand(B,T,MAX_N,MAX_N,self.cfg.num_features_relation)
            

            actor_score=self.fc_actors(edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N)
            actor_score=torch.softmax(actor_score,dim=-1).reshape(B,T,MAX_N,MAX_N) # B,T,N,N 
            # Below is the old version, it's not correct for the person-disappear cases.
            # actor_score=(actor_score+actor_score.transpose(-1,-2))/2
            
            boxes_states_pooled=torch.max((edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N,self.cfg.num_features_relation),dim=2)[0]
            acty_score=self.fc_activities(boxes_states_pooled)
            
            node_states_pooled=torch.max(edge_feature_matrix*boxes_exist_matrix_line,dim=3)[0]
            action_score= self.fc_actions(node_states_pooled)
            
            if self.training:
                return action_score,acty_score, actor_score, boxes_states_pooled.reshape(B,T,-1)  # Here, b is 3 times as the original b.
            else:
                return action_score,acty_score, actor_score  # Here, b is 3 times as the original b.

class OneGroupV5Net_wolong(nn.Module):
    """
    main module of GCN for the collective dataset
    """

    def __init__(self, cfg):
        super(OneGroupV5Net_wolong, self).__init__()
        ################ Parameters #####################
        self.cfg = cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
        MAX_N = self.cfg.num_boxes
        self.R2 = nn.Parameter(torch.tensor([_i for _i in range(1,MAX_N+1)],dtype=torch.float32), requires_grad=False)[None,:] # 1,2,3,4,...
        ################ Parameters #####################

        ################ Modules ########################
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        # if not self.cfg.train_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False
        assert self.cfg.continuous_frames>=3 and self.cfg.continuous_frames%2==1
        conv3d=[]
        for i in range((self.cfg.continuous_frames-1)//2):
            conv3d.append(BasicConv3d(D,D, kernel_size=(3,1,1),stride=1))
        self.conv3d_app=nn.Sequential(*conv3d)
        self.fc_app=nn.Linear(D*K*K,self.cfg.num_features_pose)
        self.nl_emb_1 = nn.LayerNorm([self.cfg.num_features_pose])
        self.gru_mt=nn.GRU(2+2,self.cfg.num_features_mt,num_layers=1,bidirectional=False)
        # graph update node
        self.graph=GCN_NN_Module_wolong(cfg,self.cfg.num_features_pose,self.cfg.num_features_mt)
        # output action
        self.fc_actions = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_actions)
        )
        #
        # use actions, plus relation, output which interaction, output who are the interactors.
        self.fc_actors = torch.nn.Sequential(
            nn.Linear(number_feature_relation,4*number_feature_relation),
            nn.Linear(4*number_feature_relation, 1)
        )
        self.fc_activities = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_activities)
        )
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        ################ Modules ########################
        
        
        ################ Init ###########################
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
        ################ Init ###########################
    def loadmodel(self, filepath):
        state = torch.load(filepath)
        backbone = OrderedDict()
        for key in state['backbone_state_dict'].keys():
            if 'Mixed_6' in key:
                continue
            else:
                backbone[key] = state['backbone_state_dict'][key]
        self.backbone.load_state_dict(backbone)
        print('Load model states from: ', filepath)

    def loadmodel_normal(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict'][key]
        self.load_state_dict(process_dict)
        print('Load all parameter from: ', filepath)
    @autocast()
    def forward(self, batch_data, ablation=False):
        with autocast():
            if ablation:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data  # B,T,MAX_N
            else:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data
            # B,T,F,C,H,W   B,T,MAX_N,4,F |  B,T     | B,T,MAX_N,2,F
            '''
            If in training:
            B,3T,MAX_N,D*F*K*K   B,3T,MAX_N,2,F |  B,3T      | B,3T,MAX_N,2,F
            3T: origin, positive, negative
            output of shape (seq_len, batch, num_directions * hidden_size)

            h_n of shape (num_layers * num_directions, batch, hidden_size)
            h_n.view(num_layers, num_directions, batch, hidden_size).
            '''

            # read config parameters
            B = boxes_in.shape[0]
            T = boxes_in.shape[1]
            H, W = self.cfg.image_size
            OH, OW = self.cfg.out_size
            D = self.cfg.emb_features
            K = self.cfg.crop_size[0]
            F = self.cfg.continuous_frames
            device = boxes_in.device
            if self.R2.device!=device:
                self.R2=self.R2.to(device)

            MAX_N = self.cfg.num_boxes
            number_feature_boxes = self.cfg.num_features_boxes
            num_features_gcn = self.cfg.num_features_gcn
            number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
            Agg_num=3*num_features_gcn+4 

            if self.training:
                B = B * 3
                T = T // 3
                pose_features.resize_((B, T) + pose_features.shape[2:])
                boxes_in.resize_((B, T) + boxes_in.shape[2:])
                bboxes_num_in.resize_((B, T))
                boxes_dhdw.resize_((B, T) + boxes_dhdw.shape[2:])
                if ablation:
                    actor_label.resize_((B, T,MAX_N,MAX_N))


            with torch.no_grad():
                pose_features = pose_features.reshape(B,T,MAX_N, D,F,K,K)  # B,T,MAX_N, D,F,K,K
                
                boxes_center = (boxes_in[:, :, :, :2,:] + boxes_in[:, :, :, 2:, :]) / 2 # B,T,N,2,F

                # boxes_center=boxes_center.transpose(3,4)#B,T,MAX_N,2
                boxes_dhdw=boxes_dhdw[...]#B,T,MAX_N,2,F
            
                boxes_exist=boxes_center[...,0].sum(dim=3)
                boxes_exist=boxes_exist>0 # B,T,N
                boxes_exist_int=boxes_exist.float()
                # Below is the old version, it's totally wrong
                boxes_exist_matrix=boxes_exist[:,:,:,None,None]*(boxes_exist[:,:,None,:,None]) # B,T,MAX_N,MAX_N,1
                boxes_exist_matrix_line=boxes_exist[:,:,:,None,None].expand(B,T,MAX_N,MAX_N,1) # B,T,MAX_N,MAX_N,1
                conflicts=torch.matmul(boxes_exist_int.transpose(1,2),boxes_exist_int)
                conflicts=conflicts==0 # B,N,N     main dim is the second N
                # bboxes_num_in=torch.zeros((B,),dtype=torch.int)
                temp=boxes_exist.sum(dim=1) #B,N
                temp=(temp>0).float()
                
                temp_index=temp*self.R2
                bboxes_num_in=(torch.argmax(temp_index,dim=1)+1).tolist() # B
            all_features=torch.cat([pose_features.reshape(B,T,MAX_N,D*F*K*K),boxes_center.reshape(B,T,MAX_N,2*F),boxes_dhdw.reshape(B,T,MAX_N,2*F)],dim=3)
            
            # Calculate mean node features
            mean_node_features=torch.sum(all_features*boxes_exist[...,None],dim=1,keepdim=True)/(torch.sum(boxes_exist[...,None].float(),dim=1,keepdim=True)+1e-12) # B,MAX_N,NFG
            # f=lambda x,y: torch.abs(x-y).sum(dim=-1)
            mask=~boxes_exist*temp[:,None,:]
            select=torch.nonzero(mask).tolist()
            for (b,t,box) in select:
                if box>=bboxes_num_in[b]:
                    continue
                if mask[b,t,box]:
                    other_node_index=conflicts[b,:,box]&boxes_exist[b,t]
                    have_other=other_node_index.sum()
                    if have_other:
                        # local_node_graph_features=node_graph_features[b,t]
                        other_node_features=all_features[b,t,other_node_index] # X, NFG 
                        min_distant_index=torch.argmin(
                            torch.abs(mean_node_features[b,:,box,:]-other_node_features).sum(dim=-1)) # 1
                        all_features[b,t,box].add_(other_node_features[min_distant_index]) # Copy here.
                    else:
                        all_features[b,t,box].add_(mean_node_features[b,0,box])
            
            pose_features,boxes_center,boxes_dhdw=all_features[...,:D*F*K*K],all_features[...,D*F*K*K:D*F*K*K+2*F],all_features[...,D*F*K*K+2*F:]
            
            del all_features
            del conflicts
            del mask
            del temp
            del mean_node_features
            del select
            pose_features = pose_features.reshape(B*T*MAX_N, D, F, K, K)
            pose_features=self.conv3d_app(pose_features).reshape(B*T*MAX_N, D*K*K) # aggregate F
            app_features=self.fc_app(pose_features).reshape(B,T,MAX_N,self.cfg.num_features_pose)
            app_features=self.nl_emb_1(app_features)
            app_features=torch.relu(app_features)
            
            mt_features=torch.cat([boxes_center.reshape(B,T,MAX_N,2,F),boxes_dhdw.reshape(B,T,MAX_N,2,F)],dim=3).permute(4,0,1,2,3).reshape(F,B*T*MAX_N,4) # F,B,T,MAX_N,4
            h1 = torch.empty((1, B*T*MAX_N, self.cfg.num_features_mt), device=device)
            torch.nn.init.xavier_normal_(h1)

            self.gru_mt.flatten_parameters()
            _,mt_features=self.gru_mt(mt_features,h1)
            mt_features=mt_features.reshape(B,T,MAX_N,self.cfg.num_features_mt)
            edge_feature_matrix=self.graph(app_features,mt_features) # B,T,MAX_N,MAX_N,num_features_relation
            
            # # Max pooling.
            # edge_feature_matrix=torch.max(edge_feature_matrix,dim=1,keepdim=True)[0].expand(B,T,MAX_N,MAX_N,self.cfg.num_features_relation)
            

            actor_score=self.fc_actors(edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N)
            actor_score=torch.softmax(actor_score,dim=-1).reshape(B,T,MAX_N,MAX_N) # B,T,N,N 
            # Below is the old version, it's not correct for the person-disappear cases.
            # actor_score=(actor_score+actor_score.transpose(-1,-2))/2
            
            boxes_states_pooled=torch.max((edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N,self.cfg.num_features_relation),dim=2)[0]
            acty_score=self.fc_activities(boxes_states_pooled)
            
            node_states_pooled=torch.max(edge_feature_matrix*boxes_exist_matrix_line,dim=3)[0]
            action_score= self.fc_actions(node_states_pooled)
            
            if self.training:
                return action_score,acty_score, actor_score, boxes_states_pooled.reshape(B,T,-1)  # Here, b is 3 times as the original b.
            else:
                return action_score,acty_score, actor_score  # Here, b is 3 times as the original b.

class OneGroupV5Net_actor(nn.Module):
    """
    main module of GCN for the collective dataset
    """

    def __init__(self, cfg):
        super(OneGroupV5Net_actor, self).__init__()
        ################ Parameters #####################
        self.cfg = cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
        MAX_N = self.cfg.num_boxes
        self.R2 = nn.Parameter(torch.tensor([_i for _i in range(1,MAX_N+1)],dtype=torch.float32), requires_grad=False)[None,:] # 1,2,3,4,...
        ################ Parameters #####################

        ################ Modules ########################
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        # if not self.cfg.train_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False
        assert self.cfg.continuous_frames>=3 and self.cfg.continuous_frames%2==1
        conv3d=[]
        for i in range((self.cfg.continuous_frames-1)//2):
            conv3d.append(BasicConv3d(D,D, kernel_size=(3,1,1),stride=1))
        self.conv3d_app=nn.Sequential(*conv3d)
        self.fc_app=nn.Linear(D*K*K,self.cfg.num_features_pose)
        self.nl_emb_1 = nn.LayerNorm([self.cfg.num_features_pose])
        self.gru_mt=nn.GRU(2+2,self.cfg.num_features_mt,num_layers=1,bidirectional=False)
        # graph update node
        self.graph=GCN_NN_Module(cfg,self.cfg.num_features_pose,self.cfg.num_features_mt)
        # output action
        self.fc_actions = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_actions)
        )
        #
        # use actions, plus relation, output which interaction, output who are the interactors.
        self.fc_actors = torch.nn.Sequential(
            nn.Linear(number_feature_relation,4*number_feature_relation),
            nn.Linear(4*number_feature_relation, 1)
        )
        self.fc_activities = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_activities)
        )
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        ################ Modules ########################
        
        
        ################ Init ###########################
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
        ################ Init ###########################
    def loadmodel(self, filepath):
        state = torch.load(filepath)
        backbone = OrderedDict()
        for key in state['backbone_state_dict'].keys():
            if 'Mixed_6' in key:
                continue
            else:
                backbone[key] = state['backbone_state_dict'][key]
        self.backbone.load_state_dict(backbone)
        print('Load model states from: ', filepath)

    def loadmodel_normal(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict'][key]
        self.load_state_dict(process_dict)
        print('Load all parameter from: ', filepath)
    @autocast()
    def forward(self, batch_data, ablation=False):
        with autocast():
            if ablation:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data  # B,T,MAX_N
            else:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data
            # B,T,F,C,H,W   B,T,MAX_N,4,F |  B,T     | B,T,MAX_N,2,F
            '''
            If in training:
            B,3T,MAX_N,D*F*K*K   B,3T,MAX_N,2,F |  B,3T      | B,3T,MAX_N,2,F
            3T: origin, positive, negative
            output of shape (seq_len, batch, num_directions * hidden_size)

            h_n of shape (num_layers * num_directions, batch, hidden_size)
            h_n.view(num_layers, num_directions, batch, hidden_size).
            '''

            # read config parameters
            B = boxes_in.shape[0]
            T = boxes_in.shape[1]
            H, W = self.cfg.image_size
            OH, OW = self.cfg.out_size
            D = self.cfg.emb_features
            K = self.cfg.crop_size[0]
            F = self.cfg.continuous_frames
            device = boxes_in.device
            if self.R2.device!=device:
                self.R2=self.R2.to(device)

            MAX_N = self.cfg.num_boxes
            number_feature_boxes = self.cfg.num_features_boxes
            num_features_gcn = self.cfg.num_features_gcn
            number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
            Agg_num=3*num_features_gcn+4 

            if self.training:
                B = B * 3
                T = T // 3
                pose_features.resize_((B, T) + pose_features.shape[2:])
                boxes_in.resize_((B, T) + boxes_in.shape[2:])
                bboxes_num_in.resize_((B, T))
                boxes_dhdw.resize_((B, T) + boxes_dhdw.shape[2:])
                if ablation:
                    actor_label.resize_((B, T,MAX_N,MAX_N))


            with torch.no_grad():
                pose_features = pose_features.reshape(B,T,MAX_N, D,F,K,K)  # B,T,MAX_N, D,F,K,K
                
                boxes_center = (boxes_in[:, :, :, :2,:] + boxes_in[:, :, :, 2:, :]) / 2 # B,T,N,2,F

                # boxes_center=boxes_center.transpose(3,4)#B,T,MAX_N,2
                boxes_dhdw=boxes_dhdw[...]#B,T,MAX_N,2,F
            
                boxes_exist=boxes_center[...,0].sum(dim=3)
                boxes_exist=boxes_exist>0 # B,T,N
                boxes_exist_int=boxes_exist.float()
                # Below is the old version, it's totally wrong
                boxes_exist_matrix=boxes_exist[:,:,:,None,None]*(boxes_exist[:,:,None,:,None]) # B,T,MAX_N,MAX_N,1
                boxes_exist_matrix_line=boxes_exist[:,:,:,None,None].expand(B,T,MAX_N,MAX_N,1) # B,T,MAX_N,MAX_N,1
                conflicts=torch.matmul(boxes_exist_int.transpose(1,2),boxes_exist_int)
                conflicts=conflicts==0 # B,N,N     main dim is the second N
                # bboxes_num_in=torch.zeros((B,),dtype=torch.int)
                temp=boxes_exist.sum(dim=1) #B,N
                temp=(temp>0).float()
                
                temp_index=temp*self.R2
                bboxes_num_in=(torch.argmax(temp_index,dim=1)+1).tolist() # B
            all_features=torch.cat([pose_features.reshape(B,T,MAX_N,D*F*K*K),boxes_center.reshape(B,T,MAX_N,2*F),boxes_dhdw.reshape(B,T,MAX_N,2*F)],dim=3)
            
            # Calculate mean node features
            mean_node_features=torch.sum(all_features*boxes_exist[...,None],dim=1,keepdim=True)/(torch.sum(boxes_exist[...,None].float(),dim=1,keepdim=True)+1e-12) # B,MAX_N,NFG
            # f=lambda x,y: torch.abs(x-y).sum(dim=-1)
            mask=~boxes_exist*temp[:,None,:]
            select=torch.nonzero(mask).tolist()
            for (b,t,box) in select:
                if box>=bboxes_num_in[b]:
                    continue
                if mask[b,t,box]:
                    other_node_index=conflicts[b,:,box]&boxes_exist[b,t]
                    have_other=other_node_index.sum()
                    if have_other:
                        # local_node_graph_features=node_graph_features[b,t]
                        other_node_features=all_features[b,t,other_node_index] # X, NFG 
                        min_distant_index=torch.argmin(
                            torch.abs(mean_node_features[b,:,box,:]-other_node_features).sum(dim=-1)) # 1
                        all_features[b,t,box].add_(other_node_features[min_distant_index]) # Copy here.
                    else:
                        all_features[b,t,box].add_(mean_node_features[b,0,box])
            
            pose_features,boxes_center,boxes_dhdw=all_features[...,:D*F*K*K],all_features[...,D*F*K*K:D*F*K*K+2*F],all_features[...,D*F*K*K+2*F:]
            
            del all_features
            del conflicts
            del mask
            del temp
            del mean_node_features
            del select
            pose_features = pose_features.reshape(B*T*MAX_N, D, F, K, K)
            pose_features=self.conv3d_app(pose_features).reshape(B*T*MAX_N, D*K*K) # aggregate F
            app_features=self.fc_app(pose_features).reshape(B,T,MAX_N,self.cfg.num_features_pose)
            app_features=self.nl_emb_1(app_features)
            app_features=torch.relu(app_features)
            
            mt_features=torch.cat([boxes_center.reshape(B,T,MAX_N,2,F),boxes_dhdw.reshape(B,T,MAX_N,2,F)],dim=3).permute(4,0,1,2,3).reshape(F,B*T*MAX_N,4) # F,B,T,MAX_N,4
            h1 = torch.empty((1, B*T*MAX_N, self.cfg.num_features_mt), device=device)
            torch.nn.init.xavier_normal_(h1)

            self.gru_mt.flatten_parameters()
            _,mt_features=self.gru_mt(mt_features,h1)
            mt_features=mt_features.reshape(B,T,MAX_N,self.cfg.num_features_mt)
            edge_feature_matrix=self.graph(app_features,mt_features) # B,T,MAX_N,MAX_N,num_features_relation
            
            # # Max pooling.
            # edge_feature_matrix=torch.max(edge_feature_matrix,dim=1,keepdim=True)[0].expand(B,T,MAX_N,MAX_N,self.cfg.num_features_relation)
            

            actor_score=self.fc_actors(edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N)
            actor_score=torch.softmax(actor_score,dim=-1).reshape(B,T,MAX_N,MAX_N) # B,T,N,N 
            # Below is the old version, it's not correct for the person-disappear cases.
            # actor_score=(actor_score+actor_score.transpose(-1,-2))/2
            
            boxes_states_pooled=torch.sum((edge_feature_matrix*actor_score[...,None]).reshape(B,T,MAX_N*MAX_N,self.cfg.num_features_relation),dim=2)
            acty_score=self.fc_activities(boxes_states_pooled)
            
            node_states_pooled=torch.sum(edge_feature_matrix*actor_score[...,None],dim=3)+torch.diagonal(edge_feature_matrix*boxes_exist_matrix,dim1=-3,dim2=-2).transpose(-1,-2)
            action_score= self.fc_actions(node_states_pooled)
            
            if self.training:
                return action_score,acty_score, actor_score, boxes_states_pooled.reshape(B,T,-1)  # Here, b is 3 times as the original b.
            else:
                return action_score,acty_score, actor_score  # Here, b is 3 times as the original b.

class OneGroupV5Net_woMOTrepair(nn.Module):
    """
    main module of GCN for the collective dataset
    """

    def __init__(self, cfg):
        super(OneGroupV5Net_woMOTrepair, self).__init__()
        ################ Parameters #####################
        self.cfg = cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
        MAX_N = self.cfg.num_boxes
        self.R2 = nn.Parameter(torch.tensor([_i for _i in range(1,MAX_N+1)],dtype=torch.float32), requires_grad=False)[None,:] # 1,2,3,4,...
        ################ Parameters #####################

        ################ Modules ########################
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        # if not self.cfg.train_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False
        assert self.cfg.continuous_frames>=3 and self.cfg.continuous_frames%2==1
        conv3d=[]
        for i in range((self.cfg.continuous_frames-1)//2):
            conv3d.append(BasicConv3d(D,D, kernel_size=(3,1,1),stride=1))
        self.conv3d_app=nn.Sequential(*conv3d)
        self.fc_app=nn.Linear(D*K*K,self.cfg.num_features_pose)
        self.nl_emb_1 = nn.LayerNorm([self.cfg.num_features_pose])
        self.gru_mt=nn.GRU(2+2,self.cfg.num_features_mt,num_layers=1,bidirectional=False)
        # graph update node
        self.graph=GCN_NN_Module(cfg,self.cfg.num_features_pose,self.cfg.num_features_mt)
        # output action
        self.fc_actions = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_actions)
        )
        # use actions, plus relation, output which interaction, output who are the interactors.
        self.fc_actors = torch.nn.Sequential(
            nn.Linear(number_feature_relation,4*number_feature_relation),
            nn.Linear(4*number_feature_relation, 1)
        )
        self.fc_activities = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_activities)
        )
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        ################ Modules ########################
        
        
        ################ Init ###########################
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
        ################ Init ###########################
    def loadmodel(self, filepath):
        state = torch.load(filepath)
        backbone = OrderedDict()
        for key in state['backbone_state_dict'].keys():
            if 'Mixed_6' in key:
                continue
            else:
                backbone[key] = state['backbone_state_dict'][key]
        self.backbone.load_state_dict(backbone)
        print('Load model states from: ', filepath)

    def loadmodel_normal(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict'][key]
        self.load_state_dict(process_dict)
        print('Load all parameter from: ', filepath)
    @autocast()
    def forward(self, batch_data, ablation=False):
        with autocast():
            if ablation:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data  # B,T,MAX_N
            else:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data
            # B,T,F,C,H,W   B,T,MAX_N,4,F |  B,T     | B,T,MAX_N,2,F
            '''
            If in training:
            B,3T,MAX_N,D*F*K*K   B,3T,MAX_N,2,F |  B,3T      | B,3T,MAX_N,2,F
            3T: origin, positive, negative
            output of shape (seq_len, batch, num_directions * hidden_size)

            h_n of shape (num_layers * num_directions, batch, hidden_size)
            h_n.view(num_layers, num_directions, batch, hidden_size).
            '''

            # read config parameters
            B = boxes_in.shape[0]
            T = boxes_in.shape[1]
            H, W = self.cfg.image_size
            OH, OW = self.cfg.out_size
            D = self.cfg.emb_features
            K = self.cfg.crop_size[0]
            F = self.cfg.continuous_frames
            device = boxes_in.device
            if self.R2.device!=device:
                self.R2=self.R2.to(device)

            MAX_N = self.cfg.num_boxes
            number_feature_boxes = self.cfg.num_features_boxes
            num_features_gcn = self.cfg.num_features_gcn
            number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
            Agg_num=3*num_features_gcn+4 

            if self.training:
                B = B * 3
                T = T // 3
                pose_features.resize_((B, T) + pose_features.shape[2:])
                boxes_in.resize_((B, T) + boxes_in.shape[2:])
                bboxes_num_in.resize_((B, T))
                boxes_dhdw.resize_((B, T) + boxes_dhdw.shape[2:])
                if ablation:
                    actor_label.resize_((B, T,MAX_N,MAX_N))


            with torch.no_grad():
                pose_features = pose_features.reshape(B,T,MAX_N, D,F,K,K)  # B,T,MAX_N, D,F,K,K
                
                boxes_center = (boxes_in[:, :, :, :2,:] + boxes_in[:, :, :, 2:, :]) / 2 # B,T,N,2,F

                # boxes_center=boxes_center.transpose(3,4)#B,T,MAX_N,2
                boxes_dhdw=boxes_dhdw[...]#B,T,MAX_N,2,F
            
            pose_features,boxes_center,boxes_dhdw=pose_features.reshape(B,T,MAX_N,D*F*K*K),boxes_center.reshape(B,T,MAX_N,2*F),boxes_dhdw.reshape(B,T,MAX_N,2*F)
            pose_features = pose_features.reshape(B*T*MAX_N, D, F, K, K)
            pose_features=self.conv3d_app(pose_features).reshape(B*T*MAX_N, D*K*K) # aggregate F
            app_features=self.fc_app(pose_features).reshape(B,T,MAX_N,self.cfg.num_features_pose)
            app_features=self.nl_emb_1(app_features)
            app_features=torch.relu(app_features)
            
            mt_features=torch.cat([boxes_center.reshape(B,T,MAX_N,2,F),boxes_dhdw.reshape(B,T,MAX_N,2,F)],dim=3).permute(4,0,1,2,3).reshape(F,B*T*MAX_N,4) # F,B,T,MAX_N,4
            h1 = torch.empty((1, B*T*MAX_N, self.cfg.num_features_mt), device=device)
            torch.nn.init.xavier_normal_(h1)

            self.gru_mt.flatten_parameters()
            _,mt_features=self.gru_mt(mt_features,h1)
            mt_features=mt_features.reshape(B,T,MAX_N,self.cfg.num_features_mt)
            edge_feature_matrix=self.graph(app_features,mt_features) # B,T,MAX_N,MAX_N,num_features_relation
            
            # # Max pooling.
            # edge_feature_matrix=torch.max(edge_feature_matrix,dim=1,keepdim=True)[0].expand(B,T,MAX_N,MAX_N,self.cfg.num_features_relation)
            

            actor_score=self.fc_actors(edge_feature_matrix).reshape(B,T,MAX_N*MAX_N)
            actor_score=torch.softmax(actor_score,dim=-1).reshape(B,T,MAX_N,MAX_N) # B,T,N,N 
            # Below is the old version, it's not correct for the person-disappear cases.
            # actor_score=(actor_score+actor_score.transpose(-1,-2))/2
            
            boxes_states_pooled=torch.max((edge_feature_matrix).reshape(B,T,MAX_N*MAX_N,self.cfg.num_features_relation),dim=2)[0]
            acty_score=self.fc_activities(boxes_states_pooled)
            
            node_states_pooled=torch.max(edge_feature_matrix,dim=3)[0]
            action_score= self.fc_actions(node_states_pooled)
            
            if self.training:
                return action_score,acty_score, actor_score, boxes_states_pooled.reshape(B,T,-1)  # Here, b is 3 times as the original b.
            else:
                return action_score,acty_score, actor_score  # Here, b is 3 times as the original b.


class OneGroupV5Net_wotraj(nn.Module):
    """
    main module of GCN for the collective dataset
    """

    def __init__(self, cfg):
        super(OneGroupV5Net_wotraj, self).__init__()
        ################ Parameters #####################
        self.cfg = cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
        MAX_N = self.cfg.num_boxes
        self.R2 = nn.Parameter(torch.tensor([_i for _i in range(1,MAX_N+1)],dtype=torch.float32), requires_grad=False)[None,:] # 1,2,3,4,...
        ################ Parameters #####################

        ################ Modules ########################
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        # if not self.cfg.train_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False
        assert self.cfg.continuous_frames>=3 and self.cfg.continuous_frames%2==1
        conv3d=[]
        for i in range((self.cfg.continuous_frames-1)//2):
            conv3d.append(BasicConv3d(D,D, kernel_size=(3,1,1),stride=1))
        self.conv3d_app=nn.Sequential(*conv3d)
        self.fc_app=nn.Linear(D*K*K,self.cfg.num_features_pose)
        self.nl_emb_1 = nn.LayerNorm([self.cfg.num_features_pose])
        self.gru_mt=nn.GRU(2+2,self.cfg.num_features_mt,num_layers=1,bidirectional=False)
        # graph update node
        self.graph=GCN_NN_Module_wotraj(cfg,self.cfg.num_features_pose,self.cfg.num_features_mt)
        # output action
        self.fc_actions = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_actions)
        )
        # use actions, plus relation, output which interaction, output who are the interactors.
        self.fc_actors = torch.nn.Sequential(
            nn.Linear(number_feature_relation,4*number_feature_relation),
            nn.Linear(4*number_feature_relation, 1)
        )
        self.fc_activities = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_activities)
        )
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        ################ Modules ########################
        
        
        ################ Init ###########################
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
        ################ Init ###########################
    def loadmodel(self, filepath):
        state = torch.load(filepath)
        backbone = OrderedDict()
        for key in state['backbone_state_dict'].keys():
            if 'Mixed_6' in key:
                continue
            else:
                backbone[key] = state['backbone_state_dict'][key]
        self.backbone.load_state_dict(backbone)
        print('Load model states from: ', filepath)

    def loadmodel_normal(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict'][key]
        self.load_state_dict(process_dict)
        print('Load all parameter from: ', filepath)
    @autocast()
    def forward(self, batch_data, ablation=False):
        with autocast():
            if ablation:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data  # B,T,MAX_N
            else:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data
            # B,T,F,C,H,W   B,T,MAX_N,4,F |  B,T     | B,T,MAX_N,2,F
            '''
            If in training:
            B,3T,MAX_N,D*F*K*K   B,3T,MAX_N,2,F |  B,3T      | B,3T,MAX_N,2,F
            3T: origin, positive, negative
            output of shape (seq_len, batch, num_directions * hidden_size)

            h_n of shape (num_layers * num_directions, batch, hidden_size)
            h_n.view(num_layers, num_directions, batch, hidden_size).
            '''

            # read config parameters
            B = boxes_in.shape[0]
            T = boxes_in.shape[1]
            H, W = self.cfg.image_size
            OH, OW = self.cfg.out_size
            D = self.cfg.emb_features
            K = self.cfg.crop_size[0]
            F = self.cfg.continuous_frames
            device = boxes_in.device
            if self.R2.device!=device:
                self.R2=self.R2.to(device)

            MAX_N = self.cfg.num_boxes
            number_feature_boxes = self.cfg.num_features_boxes
            num_features_gcn = self.cfg.num_features_gcn
            number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
            Agg_num=3*num_features_gcn+4 

            if self.training:
                B = B * 3
                T = T // 3
                pose_features.resize_((B, T) + pose_features.shape[2:])
                boxes_in.resize_((B, T) + boxes_in.shape[2:])
                bboxes_num_in.resize_((B, T))
                boxes_dhdw.resize_((B, T) + boxes_dhdw.shape[2:])
                if ablation:
                    actor_label.resize_((B, T,MAX_N,MAX_N))


            with torch.no_grad():
                pose_features = pose_features.reshape(B,T,MAX_N, D,F,K,K)  # B,T,MAX_N, D,F,K,K
                
                boxes_center = (boxes_in[:, :, :, :2,:] + boxes_in[:, :, :, 2:, :]) / 2 # B,T,N,2,F

                # boxes_center=boxes_center.transpose(3,4)#B,T,MAX_N,2
                boxes_dhdw=boxes_dhdw[...]#B,T,MAX_N,2,F
            
                boxes_exist=boxes_center[...,0].sum(dim=3)
                boxes_exist=boxes_exist>0 # B,T,N
                boxes_exist_int=boxes_exist.float()
                # Below is the old version, it's totally wrong
                boxes_exist_matrix=boxes_exist[:,:,:,None,None]*(boxes_exist[:,:,None,:,None]) # B,T,MAX_N,MAX_N,1
                boxes_exist_matrix_line=boxes_exist[:,:,:,None,None].expand(B,T,MAX_N,MAX_N,1) # B,T,MAX_N,MAX_N,1
                conflicts=torch.matmul(boxes_exist_int.transpose(1,2),boxes_exist_int)
                conflicts=conflicts==0 # B,N,N     main dim is the second N
                # bboxes_num_in=torch.zeros((B,),dtype=torch.int)
                temp=boxes_exist.sum(dim=1) #B,N
                temp=(temp>0).float()
                
                temp_index=temp*self.R2
                bboxes_num_in=(torch.argmax(temp_index,dim=1)+1).tolist() # B
            all_features=torch.cat([pose_features.reshape(B,T,MAX_N,D*F*K*K),boxes_center.reshape(B,T,MAX_N,2*F),boxes_dhdw.reshape(B,T,MAX_N,2*F)],dim=3)
            
            # Calculate mean node features
            mean_node_features=torch.sum(all_features*boxes_exist[...,None],dim=1,keepdim=True)/(torch.sum(boxes_exist[...,None].float(),dim=1,keepdim=True)+1e-12) # B,MAX_N,NFG
            # f=lambda x,y: torch.abs(x-y).sum(dim=-1)
            mask=~boxes_exist*temp[:,None,:]
            select=torch.nonzero(mask).tolist()
            for (b,t,box) in select:
                if box>=bboxes_num_in[b]:
                    continue
                if mask[b,t,box]:
                    other_node_index=conflicts[b,:,box]&boxes_exist[b,t]
                    have_other=other_node_index.sum()
                    if have_other:
                        # local_node_graph_features=node_graph_features[b,t]
                        other_node_features=all_features[b,t,other_node_index] # X, NFG 
                        min_distant_index=torch.argmin(
                            torch.abs(mean_node_features[b,:,box,:]-other_node_features).sum(dim=-1)) # 1
                        all_features[b,t,box].add_(other_node_features[min_distant_index]) # Copy here.
                    else:
                        all_features[b,t,box].add_(mean_node_features[b,0,box])
            
            pose_features,boxes_center,boxes_dhdw=all_features[...,:D*F*K*K],all_features[...,D*F*K*K:D*F*K*K+2*F],all_features[...,D*F*K*K+2*F:]
            
            del all_features
            del conflicts
            del mask
            del temp
            del mean_node_features
            del select
            pose_features = pose_features.reshape(B*T*MAX_N, D, F, K, K)
            pose_features=self.conv3d_app(pose_features).reshape(B*T*MAX_N, D*K*K) # aggregate F
            app_features=self.fc_app(pose_features).reshape(B,T,MAX_N,self.cfg.num_features_pose)
            app_features=self.nl_emb_1(app_features)
            app_features=torch.relu(app_features)
            
            # mt_features=torch.cat([boxes_center.reshape(B,T,MAX_N,2,F),boxes_dhdw.reshape(B,T,MAX_N,2,F)],dim=3).permute(4,0,1,2,3).reshape(F,B*T*MAX_N,4) # F,B,T,MAX_N,4
            # h1 = torch.empty((1, B*T*MAX_N, self.cfg.num_features_mt), device=device)
            # torch.nn.init.xavier_normal_(h1)

            # self.gru_mt.flatten_parameters()
            # _,mt_features=self.gru_mt(mt_features,h1)
            # mt_features=mt_features.reshape(B,T,MAX_N,self.cfg.num_features_mt)
            edge_feature_matrix=self.graph(app_features) # B,T,MAX_N,MAX_N,num_features_relation
            
            # # Max pooling.
            # edge_feature_matrix=torch.max(edge_feature_matrix,dim=1,keepdim=True)[0].expand(B,T,MAX_N,MAX_N,self.cfg.num_features_relation)
            

            actor_score=self.fc_actors(edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N)
            actor_score=torch.softmax(actor_score,dim=-1).reshape(B,T,MAX_N,MAX_N) # B,T,N,N 
            # Below is the old version, it's not correct for the person-disappear cases.
            # actor_score=(actor_score+actor_score.transpose(-1,-2))/2
            
            boxes_states_pooled=torch.max((edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N,self.cfg.num_features_relation),dim=2)[0]
            acty_score=self.fc_activities(boxes_states_pooled)
            
            node_states_pooled=torch.max(edge_feature_matrix*boxes_exist_matrix_line,dim=3)[0]
            action_score= self.fc_actions(node_states_pooled)
            
            if self.training:
                return action_score,acty_score, actor_score, boxes_states_pooled.reshape(B,T,-1)  # Here, b is 3 times as the original b.
            else:
                return action_score,acty_score, actor_score  # Here, b is 3 times as the original b.


class OneGroupV5Net_TSN(nn.Module):
    """
    main module of GCN for the collective dataset
    """

    def __init__(self, cfg):
        super(OneGroupV5Net_TSN, self).__init__()
        ################ Parameters #####################
        self.cfg = cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
        MAX_N = self.cfg.num_boxes
        self.R2 = nn.Parameter(torch.tensor([_i for _i in range(1,MAX_N+1)],dtype=torch.float32), requires_grad=False)[None,:] # 1,2,3,4,...
        ################ Parameters #####################

        ################ Modules ########################
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        # if not self.cfg.train_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False
        # assert self.cfg.continuous_frames>=3 and self.cfg.continuous_frames%2==1
        conv3d=[]
        # for i in range((self.cfg.continuous_frames-1)//2):
        conv3d.append(BasicConv3d(D,D, kernel_size=(1,1,1),stride=1))
        self.conv3d_app=nn.Sequential(*conv3d)
        self.fc_app=nn.Linear(D*K*K,self.cfg.num_features_pose)
        self.nl_emb_1 = nn.LayerNorm([self.cfg.num_features_pose])
        self.gru_mt=nn.GRU(2+2,self.cfg.num_features_mt,num_layers=1,bidirectional=False)
        # graph update node
        self.graph=GCN_NN_Module(cfg,self.cfg.num_features_pose,self.cfg.num_features_mt)
        # output action
        self.fc_actions = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_actions)
        )
        # use actions, plus relation, output which interaction, output who are the interactors.
        self.fc_actors = torch.nn.Sequential(
            nn.Linear(number_feature_relation,4*number_feature_relation),
            nn.Linear(4*number_feature_relation, 1)
        )
        self.fc_activities = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_activities)
        )
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        ################ Modules ########################
        
        
        ################ Init ###########################
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
        ################ Init ###########################
    def loadmodel(self, filepath):
        state = torch.load(filepath)
        backbone = OrderedDict()
        for key in state['backbone_state_dict'].keys():
            if 'Mixed_6' in key:
                continue
            else:
                backbone[key] = state['backbone_state_dict'][key]
        self.backbone.load_state_dict(backbone)
        print('Load model states from: ', filepath)

    def loadmodel_normal(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict'][key]
        self.load_state_dict(process_dict)
        print('Load all parameter from: ', filepath)
    @autocast()
    def forward(self, batch_data, ablation=False):
        with autocast():
            if ablation:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data  # B,T,MAX_N
            else:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data
            # B,T,F,C,H,W   B,T,MAX_N,4,F |  B,T     | B,T,MAX_N,2,F
            '''
            If in training:
            B,3T,MAX_N,D*F*K*K   B,3T,MAX_N,2,F |  B,3T      | B,3T,MAX_N,2,F
            3T: origin, positive, negative
            output of shape (seq_len, batch, num_directions * hidden_size)

            h_n of shape (num_layers * num_directions, batch, hidden_size)
            h_n.view(num_layers, num_directions, batch, hidden_size).
            '''

            # read config parameters
            B = boxes_in.shape[0]
            T = boxes_in.shape[1]
            H, W = self.cfg.image_size
            OH, OW = self.cfg.out_size
            D = self.cfg.emb_features
            K = self.cfg.crop_size[0]
            F = self.cfg.continuous_frames
            device = boxes_in.device
            if self.R2.device!=device:
                self.R2=self.R2.to(device)

            MAX_N = self.cfg.num_boxes
            number_feature_boxes = self.cfg.num_features_boxes
            num_features_gcn = self.cfg.num_features_gcn
            number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
            Agg_num=3*num_features_gcn+4 

            if self.training:
                B = B * 3
                T = T // 3
                pose_features.resize_((B, T) + pose_features.shape[2:])
                boxes_in.resize_((B, T) + boxes_in.shape[2:])
                bboxes_num_in.resize_((B, T))
                boxes_dhdw.resize_((B, T) + boxes_dhdw.shape[2:])
                if ablation:
                    actor_label.resize_((B, T,MAX_N,MAX_N))
            
            # Since TSN has no short-term sampling
            pose_features = pose_features.reshape(B,T,MAX_N, D,F,K,K)  # B,T,MAX_N, D,F,K,K
            pose_features=pose_features[...,0,None,:,:]
            F=1
            boxes_in=boxes_in[:,:,:,:,0,None]
            boxes_dhdw=boxes_dhdw[:,:,:,:,0,None]



            with torch.no_grad():
                boxes_center = (boxes_in[:, :, :, :2,:] + boxes_in[:, :, :, 2:, :]) / 2 # B,T,N,2,F

                # boxes_center=boxes_center.transpose(3,4)#B,T,MAX_N,2
                boxes_dhdw=boxes_dhdw[...]#B,T,MAX_N,2,F
            
                boxes_exist=boxes_center[...,0].sum(dim=3)
                boxes_exist=boxes_exist>0 # B,T,N
                boxes_exist_int=boxes_exist.float()
                # Below is the old version, it's totally wrong
                boxes_exist_matrix=boxes_exist[:,:,:,None,None]*(boxes_exist[:,:,None,:,None]) # B,T,MAX_N,MAX_N,1
                boxes_exist_matrix_line=boxes_exist[:,:,:,None,None].expand(B,T,MAX_N,MAX_N,1) # B,T,MAX_N,MAX_N,1
                conflicts=torch.matmul(boxes_exist_int.transpose(1,2),boxes_exist_int)
                conflicts=conflicts==0 # B,N,N     main dim is the second N
                # bboxes_num_in=torch.zeros((B,),dtype=torch.int)
                temp=boxes_exist.sum(dim=1) #B,N
                temp=(temp>0).float()
                
                temp_index=temp*self.R2
                bboxes_num_in=(torch.argmax(temp_index,dim=1)+1).tolist() # B
            all_features=torch.cat([pose_features.reshape(B,T,MAX_N,D*F*K*K),boxes_center.reshape(B,T,MAX_N,2*F),boxes_dhdw.reshape(B,T,MAX_N,2*F)],dim=3)
            
            # Calculate mean node features
            mean_node_features=torch.sum(all_features*boxes_exist[...,None],dim=1,keepdim=True)/(torch.sum(boxes_exist[...,None].float(),dim=1,keepdim=True)+1e-12) # B,MAX_N,NFG
            # f=lambda x,y: torch.abs(x-y).sum(dim=-1)
            mask=~boxes_exist*temp[:,None,:]
            select=torch.nonzero(mask).tolist()
            for (b,t,box) in select:
                if box>=bboxes_num_in[b]:
                    continue
                if mask[b,t,box]:
                    other_node_index=conflicts[b,:,box]&boxes_exist[b,t]
                    have_other=other_node_index.sum()
                    if have_other:
                        # local_node_graph_features=node_graph_features[b,t]
                        other_node_features=all_features[b,t,other_node_index] # X, NFG 
                        min_distant_index=torch.argmin(
                            torch.abs(mean_node_features[b,:,box,:]-other_node_features).sum(dim=-1)) # 1
                        all_features[b,t,box].add_(other_node_features[min_distant_index]) # Copy here.
                    else:
                        all_features[b,t,box].add_(mean_node_features[b,0,box])
            
            pose_features,boxes_center,boxes_dhdw=all_features[...,:D*F*K*K],all_features[...,D*F*K*K:D*F*K*K+2*F],all_features[...,D*F*K*K+2*F:]
            
            del all_features
            del conflicts
            del mask
            del temp
            del mean_node_features
            del select
            pose_features = pose_features.reshape(B*T*MAX_N, D, F, K, K)
            pose_features=self.conv3d_app(pose_features).reshape(B*T*MAX_N, D*K*K) # aggregate F
            app_features=self.fc_app(pose_features).reshape(B,T,MAX_N,self.cfg.num_features_pose)
            app_features=self.nl_emb_1(app_features)
            app_features=torch.relu(app_features)
            
            mt_features=torch.cat([boxes_center.reshape(B,T,MAX_N,2,F),boxes_dhdw.reshape(B,T,MAX_N,2,F)],dim=3).permute(4,0,1,2,3).reshape(F,B*T*MAX_N,4) # F,B,T,MAX_N,4
            h1 = torch.empty((1, B*T*MAX_N, self.cfg.num_features_mt), device=device)
            torch.nn.init.xavier_normal_(h1)

            self.gru_mt.flatten_parameters()
            _,mt_features=self.gru_mt(mt_features,h1)
            mt_features=mt_features.reshape(B,T,MAX_N,self.cfg.num_features_mt)
            edge_feature_matrix=self.graph(app_features,mt_features) # B,T,MAX_N,MAX_N,num_features_relation
            
            # # Max pooling.
            # edge_feature_matrix=torch.max(edge_feature_matrix,dim=1,keepdim=True)[0].expand(B,T,MAX_N,MAX_N,self.cfg.num_features_relation)
            

            actor_score=self.fc_actors(edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N)
            actor_score=torch.softmax(actor_score,dim=-1).reshape(B,T,MAX_N,MAX_N) # B,T,N,N 
            # Below is the old version, it's not correct for the person-disappear cases.
            # actor_score=(actor_score+actor_score.transpose(-1,-2))/2
            
            boxes_states_pooled=torch.max((edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N,self.cfg.num_features_relation),dim=2)[0]
            acty_score=self.fc_activities(boxes_states_pooled)
            
            node_states_pooled=torch.max(edge_feature_matrix*boxes_exist_matrix_line,dim=3)[0]
            action_score= self.fc_actions(node_states_pooled)
            
            if self.training:
                return action_score,acty_score, actor_score, boxes_states_pooled.reshape(B,T,-1)  # Here, b is 3 times as the original b.
            else:
                return action_score,acty_score, actor_score  # Here, b is 3 times as the original b.
            
class OneGroupV5Net_woapp(nn.Module):
    """
    main module of GCN for the collective dataset
    """

    def __init__(self, cfg):
        super(OneGroupV5Net_woapp, self).__init__()
        ################ Parameters #####################
        self.cfg = cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
        MAX_N = self.cfg.num_boxes
        self.R2 = nn.Parameter(torch.tensor([_i for _i in range(1,MAX_N+1)],dtype=torch.float32), requires_grad=False)[None,:] # 1,2,3,4,...
        ################ Parameters #####################

        ################ Modules ########################
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        # if not self.cfg.train_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False
        assert self.cfg.continuous_frames>=3 and self.cfg.continuous_frames%2==1
        conv3d=[]
        for i in range((self.cfg.continuous_frames-1)//2):
            conv3d.append(BasicConv3d(D,D, kernel_size=(3,1,1),stride=1))
        self.conv3d_app=nn.Sequential(*conv3d)
        self.fc_app=nn.Linear(D*K*K,self.cfg.num_features_pose)
        self.nl_emb_1 = nn.LayerNorm([self.cfg.num_features_pose])
        self.gru_mt=nn.GRU(2+2,self.cfg.num_features_mt,num_layers=1,bidirectional=False)
        # graph update node
        self.graph=GCN_NN_Module_woapp(cfg,self.cfg.num_features_pose,self.cfg.num_features_mt)
        # output action
        self.fc_actions = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_actions)
        )
        #
        # use actions, plus relation, output which interaction, output who are the interactors.
        self.fc_actors = torch.nn.Sequential(
            nn.Linear(number_feature_relation,4*number_feature_relation),
            nn.Linear(4*number_feature_relation, 1)
        )
        self.fc_activities = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_activities)
        )
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        ################ Modules ########################
        
        
        ################ Init ###########################
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
        ################ Init ###########################
    def loadmodel(self, filepath):
        state = torch.load(filepath)
        backbone = OrderedDict()
        for key in state['backbone_state_dict'].keys():
            if 'Mixed_6' in key:
                continue
            else:
                backbone[key] = state['backbone_state_dict'][key]
        self.backbone.load_state_dict(backbone)
        print('Load model states from: ', filepath)

    def loadmodel_normal(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict'][key]
        self.load_state_dict(process_dict)
        print('Load all parameter from: ', filepath)
    @autocast()
    def forward(self, batch_data, ablation=False):
        with autocast():
            if ablation:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data  # B,T,MAX_N
            else:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data
            # B,T,F,C,H,W   B,T,MAX_N,4,F |  B,T     | B,T,MAX_N,2,F
            '''
            If in training:
            B,3T,MAX_N,D*F*K*K   B,3T,MAX_N,2,F |  B,3T      | B,3T,MAX_N,2,F
            3T: origin, positive, negative
            output of shape (seq_len, batch, num_directions * hidden_size)

            h_n of shape (num_layers * num_directions, batch, hidden_size)
            h_n.view(num_layers, num_directions, batch, hidden_size).
            '''

            # read config parameters
            B = boxes_in.shape[0]
            T = boxes_in.shape[1]
            H, W = self.cfg.image_size
            OH, OW = self.cfg.out_size
            D = self.cfg.emb_features
            K = self.cfg.crop_size[0]
            F = self.cfg.continuous_frames
            device = boxes_in.device
            if self.R2.device!=device:
                self.R2=self.R2.to(device)

            MAX_N = self.cfg.num_boxes
            number_feature_boxes = self.cfg.num_features_boxes
            num_features_gcn = self.cfg.num_features_gcn
            number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
            Agg_num=3*num_features_gcn+4 

            if self.training:
                B = B * 3
                T = T // 3
                pose_features.resize_((B, T) + pose_features.shape[2:])
                boxes_in.resize_((B, T) + boxes_in.shape[2:])
                bboxes_num_in.resize_((B, T))
                boxes_dhdw.resize_((B, T) + boxes_dhdw.shape[2:])
                if ablation:
                    actor_label.resize_((B, T,MAX_N,MAX_N))


            with torch.no_grad():
                pose_features = pose_features.reshape(B,T,MAX_N, D,F,K,K)  # B,T,MAX_N, D,F,K,K
                
                boxes_center = (boxes_in[:, :, :, :2,:] + boxes_in[:, :, :, 2:, :]) / 2 # B,T,N,2,F

                # boxes_center=boxes_center.transpose(3,4)#B,T,MAX_N,2
                boxes_dhdw=boxes_dhdw[...]#B,T,MAX_N,2,F
            
                boxes_exist=boxes_center[...,0].sum(dim=3)
                boxes_exist=boxes_exist>0 # B,T,N
                boxes_exist_int=boxes_exist.float()
                # Below is the old version, it's totally wrong
                boxes_exist_matrix=boxes_exist[:,:,:,None,None]*(boxes_exist[:,:,None,:,None]) # B,T,MAX_N,MAX_N,1
                boxes_exist_matrix_line=boxes_exist[:,:,:,None,None].expand(B,T,MAX_N,MAX_N,1) # B,T,MAX_N,MAX_N,1
                conflicts=torch.matmul(boxes_exist_int.transpose(1,2),boxes_exist_int)
                conflicts=conflicts==0 # B,N,N     main dim is the second N
                # bboxes_num_in=torch.zeros((B,),dtype=torch.int)
                temp=boxes_exist.sum(dim=1) #B,N
                temp=(temp>0).float()
                
                temp_index=temp*self.R2
                bboxes_num_in=(torch.argmax(temp_index,dim=1)+1).tolist() # B
            all_features=torch.cat([pose_features.reshape(B,T,MAX_N,D*F*K*K),boxes_center.reshape(B,T,MAX_N,2*F),boxes_dhdw.reshape(B,T,MAX_N,2*F)],dim=3)
            
            # Calculate mean node features
            mean_node_features=torch.sum(all_features*boxes_exist[...,None],dim=1,keepdim=True)/(torch.sum(boxes_exist[...,None].float(),dim=1,keepdim=True)+1e-12) # B,MAX_N,NFG
            # f=lambda x,y: torch.abs(x-y).sum(dim=-1)
            mask=~boxes_exist*temp[:,None,:]
            select=torch.nonzero(mask).tolist()
            for (b,t,box) in select:
                if box>=bboxes_num_in[b]:
                    continue
                if mask[b,t,box]:
                    other_node_index=conflicts[b,:,box]&boxes_exist[b,t]
                    have_other=other_node_index.sum()
                    if have_other:
                        # local_node_graph_features=node_graph_features[b,t]
                        other_node_features=all_features[b,t,other_node_index] # X, NFG 
                        min_distant_index=torch.argmin(
                            torch.abs(mean_node_features[b,:,box,:]-other_node_features).sum(dim=-1)) # 1
                        all_features[b,t,box].add_(other_node_features[min_distant_index]) # Copy here.
                    else:
                        all_features[b,t,box].add_(mean_node_features[b,0,box])
            
            pose_features,boxes_center,boxes_dhdw=all_features[...,:D*F*K*K],all_features[...,D*F*K*K:D*F*K*K+2*F],all_features[...,D*F*K*K+2*F:]
            
            del all_features
            del conflicts
            del mask
            del temp
            del mean_node_features
            del select
            pose_features = pose_features.reshape(B*T*MAX_N, D, F, K, K)
            pose_features=self.conv3d_app(pose_features).reshape(B*T*MAX_N, D*K*K) # aggregate F
            app_features=self.fc_app(pose_features).reshape(B,T,MAX_N,self.cfg.num_features_pose)
            app_features=self.nl_emb_1(app_features)
            app_features=torch.relu(app_features)
            
            mt_features=torch.cat([boxes_center.reshape(B,T,MAX_N,2,F),boxes_dhdw.reshape(B,T,MAX_N,2,F)],dim=3).permute(4,0,1,2,3).reshape(F,B*T*MAX_N,4) # F,B,T,MAX_N,4
            h1 = torch.empty((1, B*T*MAX_N, self.cfg.num_features_mt), device=device)
            torch.nn.init.xavier_normal_(h1)

            self.gru_mt.flatten_parameters()
            _,mt_features=self.gru_mt(mt_features,h1)
            mt_features=mt_features.reshape(B,T,MAX_N,self.cfg.num_features_mt)
            edge_feature_matrix=self.graph(mt_features) # B,T,MAX_N,MAX_N,num_features_relation
            
            # # Max pooling.
            # edge_feature_matrix=torch.max(edge_feature_matrix,dim=1,keepdim=True)[0].expand(B,T,MAX_N,MAX_N,self.cfg.num_features_relation)
            

            actor_score=self.fc_actors(edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N)
            actor_score=torch.softmax(actor_score,dim=-1).reshape(B,T,MAX_N,MAX_N) # B,T,N,N 
            # Below is the old version, it's not correct for the person-disappear cases.
            # actor_score=(actor_score+actor_score.transpose(-1,-2))/2
            
            boxes_states_pooled=torch.max((edge_feature_matrix*boxes_exist_matrix).reshape(B,T,MAX_N*MAX_N,self.cfg.num_features_relation),dim=2)[0]
            acty_score=self.fc_activities(boxes_states_pooled)
            
            node_states_pooled=torch.max(edge_feature_matrix*boxes_exist_matrix_line,dim=3)[0]
            action_score= self.fc_actions(node_states_pooled)
            
            if self.training:
                return action_score,acty_score, actor_score, boxes_states_pooled.reshape(B,T,-1)  # Here, b is 3 times as the original b.
            else:
                return action_score,acty_score, actor_score  # Here, b is 3 times as the original b.
            
            
class OneGroupV5Net_GCN(nn.Module):
    """
    main module of GCN for the collective dataset
    """

    def __init__(self, cfg):
        super(OneGroupV5Net_GCN, self).__init__()
        ################ Parameters #####################
        self.cfg = cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
        MAX_N = self.cfg.num_boxes
        self.R2 = nn.Parameter(torch.tensor([_i for _i in range(1,MAX_N+1)],dtype=torch.float32), requires_grad=False)[None,:] # 1,2,3,4,...
        ################ Parameters #####################

        ################ Modules ########################
        # self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        # if not self.cfg.train_backbone:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False
        assert self.cfg.continuous_frames>=3 and self.cfg.continuous_frames%2==1
        conv3d=[]
        for i in range((self.cfg.continuous_frames-1)//2):
            conv3d.append(BasicConv3d(D,D, kernel_size=(3,1,1),stride=1))
        self.conv3d_app=nn.Sequential(*conv3d)
        self.fc_app=nn.Linear(D*K*K,self.cfg.num_features_pose)
        self.nl_emb_1 = nn.LayerNorm([self.cfg.num_features_pose])
        self.gru_mt=nn.GRU(2+2,self.cfg.num_features_mt,num_layers=1,bidirectional=False)
        # graph update node
        self.graph=GCN_ARG_Module(cfg,self.cfg.num_features_pose+self.cfg.num_features_mt)
        # output action
        self.fc_actions = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_actions)
        )
        #
        # use actions, plus relation, output which interaction, output who are the interactors.
        self.fc_actors = torch.nn.Sequential(
            nn.Linear(number_feature_relation,4*number_feature_relation),
            nn.Linear(4*number_feature_relation, 1)
        )
        self.fc_activities = nn.Sequential(
            nn.Linear(number_feature_relation, 4*number_feature_relation),
            nn.Linear(4*number_feature_relation, self.cfg.num_activities)
        )
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        ################ Modules ########################
        
        
        ################ Init ###########################
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
        ################ Init ###########################
    def loadmodel(self, filepath):
        state = torch.load(filepath)
        backbone = OrderedDict()
        for key in state['backbone_state_dict'].keys():
            if 'Mixed_6' in key:
                continue
            else:
                backbone[key] = state['backbone_state_dict'][key]
        self.backbone.load_state_dict(backbone)
        print('Load model states from: ', filepath)

    def loadmodel_normal(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict'][key]
        self.load_state_dict(process_dict)
        print('Load all parameter from: ', filepath)
    @autocast()
    def forward(self, batch_data, ablation=False):
        with autocast():
            if ablation:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data  # B,T,MAX_N
            else:
                pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data
            # B,T,F,C,H,W   B,T,MAX_N,4,F |  B,T     | B,T,MAX_N,2,F
            '''
            If in training:
            B,3T,MAX_N,D*F*K*K   B,3T,MAX_N,2,F |  B,3T      | B,3T,MAX_N,2,F
            3T: origin, positive, negative
            output of shape (seq_len, batch, num_directions * hidden_size)

            h_n of shape (num_layers * num_directions, batch, hidden_size)
            h_n.view(num_layers, num_directions, batch, hidden_size).
            '''

            # read config parameters
            B = boxes_in.shape[0]
            T = boxes_in.shape[1]
            H, W = self.cfg.image_size
            OH, OW = self.cfg.out_size
            D = self.cfg.emb_features
            K = self.cfg.crop_size[0]
            F = self.cfg.continuous_frames
            device = boxes_in.device
            if self.R2.device!=device:
                self.R2=self.R2.to(device)

            MAX_N = self.cfg.num_boxes
            number_feature_boxes = self.cfg.num_features_boxes
            num_features_gcn = self.cfg.num_features_gcn
            number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
            Agg_num=3*num_features_gcn+4 

            if self.training:
                B = B * 3
                T = T // 3
                pose_features.resize_((B, T) + pose_features.shape[2:])
                boxes_in.resize_((B, T) + boxes_in.shape[2:])
                bboxes_num_in.resize_((B, T))
                boxes_dhdw.resize_((B, T) + boxes_dhdw.shape[2:])
                if ablation:
                    actor_label.resize_((B, T,MAX_N,MAX_N))


            with torch.no_grad():
                pose_features = pose_features.reshape(B,T,MAX_N, D,F,K,K)  # B,T,MAX_N, D,F,K,K
                
                boxes_center = (boxes_in[:, :, :, :2,:] + boxes_in[:, :, :, 2:, :]) / 2 # B,T,N,2,F

                # boxes_center=boxes_center.transpose(3,4)#B,T,MAX_N,2
                boxes_dhdw=boxes_dhdw[...]#B,T,MAX_N,2,F
            
                boxes_exist=boxes_center[...,0].sum(dim=3)
                boxes_exist=boxes_exist>0 # B,T,N
                boxes_exist_int=boxes_exist.float()
                # Below is the old version, it's totally wrong
                boxes_exist_matrix=boxes_exist[:,:,:,None,None]*(boxes_exist[:,:,None,:,None]) # B,T,MAX_N,MAX_N,1
                boxes_exist_matrix_line=boxes_exist[:,:,:,None,None].expand(B,T,MAX_N,MAX_N,1) # B,T,MAX_N,MAX_N,1
                conflicts=torch.matmul(boxes_exist_int.transpose(1,2),boxes_exist_int)
                conflicts=conflicts==0 # B,N,N     main dim is the second N
                # bboxes_num_in=torch.zeros((B,),dtype=torch.int)
                temp=boxes_exist.sum(dim=1) #B,N
                temp=(temp>0).float()
                
                temp_index=temp*self.R2
                bboxes_num_in=(torch.argmax(temp_index,dim=1)+1).tolist() # B
            all_features=torch.cat([pose_features.reshape(B,T,MAX_N,D*F*K*K),boxes_center.reshape(B,T,MAX_N,2*F),boxes_dhdw.reshape(B,T,MAX_N,2*F)],dim=3)
            
            # Calculate mean node features
            mean_node_features=torch.sum(all_features*boxes_exist[...,None],dim=1,keepdim=True)/(torch.sum(boxes_exist[...,None].float(),dim=1,keepdim=True)+1e-12) # B,MAX_N,NFG
            # f=lambda x,y: torch.abs(x-y).sum(dim=-1)
            mask=~boxes_exist*temp[:,None,:]
            select=torch.nonzero(mask).tolist()
            for (b,t,box) in select:
                if box>=bboxes_num_in[b]:
                    continue
                if mask[b,t,box]:
                    other_node_index=conflicts[b,:,box]&boxes_exist[b,t]
                    have_other=other_node_index.sum()
                    if have_other:
                        # local_node_graph_features=node_graph_features[b,t]
                        other_node_features=all_features[b,t,other_node_index] # X, NFG 
                        min_distant_index=torch.argmin(
                            torch.abs(mean_node_features[b,:,box,:]-other_node_features).sum(dim=-1)) # 1
                        all_features[b,t,box].add_(other_node_features[min_distant_index]) # Copy here.
                    else:
                        all_features[b,t,box].add_(mean_node_features[b,0,box])
            
            pose_features,boxes_center,boxes_dhdw=all_features[...,:D*F*K*K],all_features[...,D*F*K*K:D*F*K*K+2*F],all_features[...,D*F*K*K+2*F:]
            
            del all_features
            del conflicts
            del mask
            del temp
            del mean_node_features
            del select
            pose_features = pose_features.reshape(B*T*MAX_N, D, F, K, K)
            pose_features=self.conv3d_app(pose_features).reshape(B*T*MAX_N, D*K*K) # aggregate F
            app_features=self.fc_app(pose_features).reshape(B,T,MAX_N,self.cfg.num_features_pose)
            app_features=self.nl_emb_1(app_features)
            app_features=torch.relu(app_features)
            
            h1 = torch.empty((1, B*T*MAX_N, self.cfg.num_features_mt), device=device)
            torch.nn.init.xavier_normal_(h1)

            self.gru_mt.flatten_parameters()
            # dis=boxes_center.reshape(B,T,MAX_N,2,F)[...,0]
            # dis=((dis[:,:,:,None,:].expand(B,T,MAX_N,MAX_N,2)-dis[:,:,None,:,:].expand(B,T,MAX_N,MAX_N,2))**2).sum(dim=-1,keepdim=True).transpose(0,1).reshape(T,B*MAX_N*MAX_N,1)
            mt_features=torch.cat([boxes_center.reshape(B,T,MAX_N,2,F),boxes_dhdw.reshape(B,T,MAX_N,2,F)],dim=3).permute(4,0,1,2,3).reshape(F,B*T*MAX_N,4) # F,B,T,MAX_N,4
            _,mt_features=self.gru_mt(mt_features,h1)
            mt_features=mt_features.reshape(B,T,MAX_N,self.cfg.num_features_mt)
            boxes_features, relation_graphs=self.graph(torch.cat([app_features,mt_features],dim=-1))
            
            # # Max pooling.
            # edge_feature_matrix=torch.max(edge_feature_matrix,dim=1,keepdim=True)[0].expand(B,T,MAX_N,MAX_N,self.cfg.num_features_relation)
            

            actor_score=relation_graphs.reshape(B,T,MAX_N*MAX_N)
            actor_score=torch.softmax(actor_score,dim=-1).reshape(B,T,MAX_N,MAX_N) # B,T,N,N 
            if True in torch.isnan(actor_score):
                print('debug')
            # Below is the old version, it's not correct for the person-disappear cases.
            # actor_score=(actor_score+actor_score.transpose(-1,-2))/2
            
            boxes_states_pooled=torch.mean(boxes_features*actor_score.sum(dim=-1,keepdim=True),dim=2).reshape(B,T,self.cfg.num_features_relation)
            acty_score=self.fc_activities(boxes_states_pooled)
            
            action_score= self.fc_actions(boxes_features)
            
            if self.training:
                return action_score,acty_score, actor_score, boxes_states_pooled.reshape(B,T,-1)  # Here, b is 3 times as the original b.
            else:
                return action_score,acty_score, actor_score  # Here, b is 3 times as the original b.