import os
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast

import numpy as np

from backbone import *
from utils import *
from collections import OrderedDict
from models_hoi.GCN_spatial_subnet import GCN_spatial_subnet
class Edge_Classifier(nn.Module):
    def __init__(self, out_dim, drop,
                    num_classes):
        super(Edge_Classifier, self).__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.drop = drop
        
        self.classifier = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.Dropout(self.drop),
            nn.ReLU(), 
            nn.Linear(self.out_dim, self.out_dim),
            nn.Dropout(self.drop),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim),
            nn.Dropout(self.drop),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.num_classes),
        )

    def forward(self, input):
        output = self.classifier(input)
        return output
class LIGHTEN_frame_level(nn.Module):
    def __init__(self, spatial_subnet='', dropout=0.0):
        super(LIGHTEN_frame_level, self).__init__()
        self.subact_classes = 10
        self.afford_classes = 12
        self.res_feat_dim = 2048
        self.num_frames = 20
        self.num_nodes = 6
        self.dropout = dropout
        self.spatial_subnet_type = spatial_subnet

        if self.spatial_subnet_type == 'GCN':
            self.preprocess_dim = 1000
            self.hidden_dim = 512
            self.out_dim = 512

        self.preprocess_human = nn.Linear(self.res_feat_dim, self.preprocess_dim)
        self.preprocess_object = nn.Linear(self.res_feat_dim, self.preprocess_dim)

        if self.spatial_subnet_type == 'GCN':
            self.in_dim = 8 + 2*self.preprocess_dim + 10
            # Initialization to Adjacency matrix
            self.A = np.zeros((6, 6))
            self.A[0, :] = 1
            self.A[:, 0] = 1
            for i in range(6):
                self.A[i, i] = 1
            self.spatial_subnet = GCN_spatial_subnet(self.in_dim, self.hidden_dim, self.out_dim, self.A)

        # RNN blocks for frame-level temporal subnet
        self.subact_frame_RNN = nn.RNN(input_size=self.out_dim, hidden_size=self.out_dim//2, num_layers=2, batch_first=True, bidirectional=True)
        self.afford_frame_RNN = nn.RNN(input_size=2*self.out_dim, hidden_size=self.out_dim, num_layers=2, batch_first=True, bidirectional=True)

        self.classifier_human = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim), 
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.subact_classes)
        )
        self.classifier_object = nn.Sequential(
            nn.Linear(2*self.out_dim, self.out_dim), 
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.afford_classes)
        )
    
    def forward(self, inputs, out_type='scores'):
        # Graph node feature initialization
        one_hot_encodings, node_bboxes, num_objs, res_features = inputs[0], inputs[1], inputs[2], inputs[3]
        batchSize = node_bboxes.shape[0]

        if self.spatial_subnet_type == 'GCN':
            node_feats = torch.zeros((batchSize, 6, 20, self.in_dim)).float().cuda()
            
            node_feats[:, 0, :, :4] = node_bboxes[:, 0, :, :]

            human_precomp_feats = torch.flatten(res_features[:, 0, :, :, :, :].reshape(batchSize*self.num_frames, self.res_feat_dim, 1, 1), 1)
            node_feats[:, 0, :, 4:4+self.preprocess_dim] = self.preprocess_human(human_precomp_feats).reshape(batchSize, self.num_frames, self.preprocess_dim)
           
            obj_precomp_feats = torch.flatten(res_features[:, 1:, :, :, :, :].reshape(batchSize*5*self.num_frames, self.res_feat_dim, 1, 1), 1)
            node_feats[:, 1:, :, 4+self.preprocess_dim:4+2*self.preprocess_dim] = self.preprocess_object(obj_precomp_feats).reshape(batchSize, 5, self.num_frames, self.preprocess_dim)

            node_feats[:, 1:, :, 4+2*self.preprocess_dim:8+2*self.preprocess_dim] = node_bboxes[:, 1:, :, :]
            node_feats[:, 1:, :, 8+2*self.preprocess_dim:] = one_hot_encodings[:, :, :, :]

            node_feats = node_feats.permute(0, 2, 1, 3)
            node_feats = node_feats[:, :, :, :].reshape(batchSize*self.num_frames, 1, 6, self.in_dim)

        # Spatial Subnet
        # Output of spatial subnet:  (batchSize, self.num_frames, 6, self.out_dim)
        if self.spatial_subnet_type == 'GCN':
            spatial_graph = self.spatial_subnet(node_feats.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(batchSize, self.num_frames, 6, self.out_dim)

        # Append human feats to object nodes
        human_node_feats = spatial_graph[:, :, 0, :]
                
        total_obj = sum(num_objs)
        count = 0
        obj_node_feats = []
        for b in range(batchSize):
            obj_feats = spatial_graph[b, :, 1: 1+num_objs[b], :]

            concat_feats = torch.zeros((self.num_frames, num_objs[b], 2*self.out_dim)).float().cuda()
            for o in range(num_objs[b]):
                concat_feats[:, o, :] = torch.cat((human_node_feats[b, :, :], obj_feats[:, o, :]), 1)

            obj_node_feats.append(concat_feats)
            
        obj_node_feats = torch.cat(obj_node_feats, dim=1)
        obj_node_feats = obj_node_feats.permute(1, 0, 2) #total_obj x num_frames x 128

        ## Frame-level Temporal subnet
        human_rnn_feats = self.subact_frame_RNN(human_node_feats, None)[0]
        obj_rnn_feats = self.afford_frame_RNN(obj_node_feats, None)[0]

        subact_cls_scores = torch.sum(self.classifier_human(human_rnn_feats), dim=1)
        afford_cls_scores = torch.sum(self.classifier_object(obj_rnn_feats), dim=1)

        if out_type == 'scores': 
            return subact_cls_scores, afford_cls_scores
        elif out_type == 'seg_feats':
            return human_rnn_feats, obj_rnn_feats

import os,sys
import numpy  as np 
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.ops import roi_align as roi_align_t
from distant_utils import generate_actors_from_action
class Distant_hoi(nn.Module):
    """
    main module of base model for collective dataset
    """

    def __init__(self, cfg):
        super(Distant_hoi, self).__init__()
        self.cfg = cfg

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        F=self.cfg.continuous_frames
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
        MAX_N = self.cfg.num_boxes

        self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        #   self.backbone=MyVGG16(pretrained=True)

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.R2 = nn.Parameter(torch.tensor([_i for _i in range(1,MAX_N+1)],dtype=torch.float32), requires_grad=False)[None,:] # 1,2,3,4,...

        # self.roi_align = RoIAlign(*self.cfg.crop_size)
        state_dim=256 #D*F*K*K
        self.state_dim = state_dim
        self.compress_fc=nn.Linear(D*F*K*K,state_dim)

        self.drop = 0#self.cfg.train_dropout_prob
        self.action_classes=7
        self.spatial_subnet_type = 'GCN'

        if self.spatial_subnet_type == 'GCN':
            self.preprocess_dim = state_dim
            self.hidden_dim = 128
            self.out_dim = 128


        if self.spatial_subnet_type == 'GCN':
            self.in_dim = 4 + self.preprocess_dim
            # Initialization to Adjacency matrix
            self.A = np.ones((MAX_N, MAX_N))
            self.spatial_subnet = GCN_spatial_subnet(self.in_dim, self.hidden_dim, self.out_dim, self.A)

        # RNN blocks for frame-level temporal subnet
        self.frame_RNN = nn.RNN(input_size=self.out_dim, hidden_size=self.out_dim//2, num_layers=2, bidirectional=True)
        self.action_classifier = Edge_Classifier(self.out_dim, self.drop, self.action_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def savemodel(self, filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
        }

        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        backbone = OrderedDict()
        for key in state['backbone_state_dict'].keys():
            if 'Mixed_6' in key:
                continue
            else:
                backbone[key] = state['backbone_state_dict'][key]
        self.backbone.load_state_dict(backbone)
        # self.emb_3d_conv.load_state_dict(state['emb_3d_conv'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        self.fc_actions.load_state_dict(state['fc_actions_state_dict'])
        self.fc_activities.load_state_dict(state['fc_activities_state_dict'])
        print('Load model states from: ', filepath)

    @autocast()
    def forward(self, batch_data):
        with autocast(enabled=False):
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
                bboxes_num_in=(torch.argmax(temp_index,dim=1)+1).tolist() 
            all_features=pose_features.reshape(B,T,MAX_N,D*F*K*K)
            
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
            
            pose_features=all_features
            
            del conflicts
            del mask
            del temp
            del mean_node_features
            del select

            pose_features= self.compress_fc(pose_features.reshape(B*T,MAX_N,-1)) # B*T,MAX_N,state_dim

            node_num = MAX_N

            node_bboxes = pose_features

            batch_human_boxes=boxes_in[...,0].reshape(B*T,MAX_N,4)

            node_feats=torch.cat([node_bboxes,batch_human_boxes],dim=-1)

            input_graph = node_feats.reshape(B, T, MAX_N, self.in_dim).permute(0,3,1,2)
            gcn_output = self.spatial_subnet(input_graph).permute(2,0,3,1)
            edge_feats = gcn_output.reshape(T,B*MAX_N,self.out_dim)

            obj_node_feats = edge_feats#.permute(1, 0, 2) #total_obj x num_frames x 128
            self.frame_RNN.flatten_parameters()
            ## Frame-level Temporal subnet
            human_rnn_feats = self.frame_RNN(obj_node_feats, None)[0] # T,B,MAX_N,self.out_dim
            
            action_cls_scores = self.action_classifier(human_rnn_feats).transpose(0,1) # B,T,MAX_N,7

            # Below, let's translate their results into our output!
            action_score=action_cls_scores.reshape(B,T,MAX_N,7)
            actor_score=1-action_score[...,0] # B,T,MAX_N
            actor_score=generate_actors_from_action(actor_score.reshape(B*T,MAX_N)).reshape(B,T,MAX_N,MAX_N) # B,T,MAX_N,MAX_N
            acty_score=torch.softmax((action_score*actor_score.sum(dim=-1)[...,None]).sum(dim=2)[:,:,1:],dim=-1) # B,T,6
            return action_score,acty_score, actor_score