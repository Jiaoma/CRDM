import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.cuda.amp import autocast

import numpy as np

from backbone import *
from utils import *
# from roi_align.roi_align import RoIAlign      # RoIAlign module
# from roi_align.roi_align import CropAndResize # crop_and_resize module
from collections import OrderedDict
from torchvision.ops import roi_align as roi_align_t

class Basenet_volleyball(nn.Module):
    """
    main module of base model for the volleyball
    """
    def __init__(self, cfg):
        super(Basenet_volleyball, self).__init__()
        self.cfg=cfg
        
        NFB=self.cfg.num_features_boxes
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        

        if cfg.backbone=='inv3':
            self.backbone=MyInception_v3(transform_input=False,pretrained=True)
        elif cfg.backbone=='vgg16':
            self.backbone=MyVGG16(pretrained=True)
        elif cfg.backbone=='vgg19':
            self.backbone=MyVGG19(pretrained=True)
        else:
            assert False
        
        self.roi_align=RoIAlign(*self.cfg.crop_size)
        
        
        self.fc_emb = nn.Linear(K*K*D,NFB)
        self.dropout_emb = nn.Dropout(p=self.cfg.train_dropout_prob)
        
        self.fc_actions=nn.Linear(NFB,self.cfg.num_actions)
        self.fc_activities=nn.Linear(NFB,self.cfg.num_activities)
        
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


    def savemodel(self,filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'fc_emb_state_dict':self.fc_emb.state_dict(),
            'fc_actions_state_dict':self.fc_actions.state_dict(),
            'fc_activities_state_dict':self.fc_activities.state_dict()
        }
        
        torch.save(state, filepath)
        print('model saved to:',filepath)

    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb.load_state_dict(state['fc_emb_state_dict'])
        self.fc_actions.load_state_dict(state['fc_actions_state_dict'])
        self.fc_activities.load_state_dict(state['fc_activities_state_dict'])
        print('Load model states from: ',filepath)

    def forward(self,batch_data):
        images_in, boxes_in = batch_data
        
        # read config parameters
        B=images_in.shape[0]
        T=images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        N=self.cfg.num_boxes
        NFB=self.cfg.num_features_boxes
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in_flat=torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4

        boxes_idx=[i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,
        
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat=prep_images(images_in_flat)
        
        outputs=self.backbone(images_in_flat)
        
        
        # Build multiscale features
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        
        
        
        # ActNet
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
#         features_multiscale.requires_grad=False
        
    
        # RoI Align
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,
        
        
        boxes_features=boxes_features.reshape(B*T*N,-1) # B*T*N, D*K*K
        
            
        # Embedding to hidden state
        boxes_features=self.fc_emb(boxes_features)  # B*T*N, NFB
        boxes_features=F.relu(boxes_features)
        boxes_features=self.dropout_emb(boxes_features)
       
    
        boxes_states=boxes_features.reshape(B,T,N,NFB)
        
        # Predict actions
        boxes_states_flat=boxes_states.reshape(-1,NFB)  #B*T*N, NFB

        actions_scores=self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        
        
        # Predict activities
        boxes_states_pooled,_=torch.max(boxes_states,dim=2)  #B, T, NFB
        boxes_states_pooled_flat=boxes_states_pooled.reshape(-1,NFB)  #B*T, NFB
        
        activities_scores=self.fc_activities(boxes_states_pooled_flat)  #B*T, acty_num
        
        if T!=1:
            actions_scores=actions_scores.reshape(B,T,N,-1).mean(dim=1).reshape(B*N,-1)
            activities_scores=activities_scores.reshape(B,T,-1).mean(dim=1)
            
        return actions_scores, activities_scores
        
        
class Basenet_collective(nn.Module):
    """
    main module of base model for collective dataset
    """
    def __init__(self, cfg):
        super(Basenet_collective, self).__init__()
        self.cfg=cfg
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        
        self.backbone=MyInception_v3(transform_input=False,pretrained=True)
#         self.backbone=MyVGG16(pretrained=True)
        
        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        
        self.roi_align=RoIAlign(*self.cfg.crop_size)
        
        self.fc_emb_1=nn.Linear(K*K*D,NFB)
        self.dropout_emb_1 = nn.Dropout(p=self.cfg.train_dropout_prob)
#         self.nl_emb_1=nn.LayerNorm([NFB])
        
        
        self.fc_actions=nn.Linear(NFB,self.cfg.num_actions)
        self.fc_activities=nn.Linear(NFB,self.cfg.num_activities)
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def savemodel(self,filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'fc_emb_state_dict':self.fc_emb_1.state_dict(),
            'fc_actions_state_dict':self.fc_actions.state_dict(),
            'fc_activities_state_dict':self.fc_activities.state_dict()
        }
        
        torch.save(state, filepath)
        print('model saved to:',filepath)
        

    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ',filepath)
        
                
    def forward(self,batch_data):
        images_in, boxes_in, bboxes_num_in = batch_data
    
        # read config parameters
        B=images_in.shape[0]
        T=images_in.shape[1]
        H, W=self.cfg.image_size
        OH, OW=self.cfg.out_size
        MAX_N=self.cfg.num_boxes
        NFB=self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        EPS=1e-5
        
        D=self.cfg.emb_features
        K=self.cfg.crop_size[0]
        
        # Reshape the input data
        images_in_flat=torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in=boxes_in.reshape(B*T,MAX_N,4)
                
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat=prep_images(images_in_flat)
        outputs=self.backbone(images_in_flat)
            
        
        # Build multiscale features
        features_multiscale=[]
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features,size=(OH,OW),mode='bilinear',align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale,dim=1)  #B*T, D, OH, OW
        

        boxes_in_flat=torch.reshape(boxes_in,(B*T*MAX_N,4))  #B*T*MAX_N, 4
            
        boxes_idx=[i * torch.ones(MAX_N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx=torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat=torch.reshape(boxes_idx,(B*T*MAX_N,))  #B*T*MAX_N,

        # RoI Align
        boxes_in_flat.requires_grad=False
        boxes_idx_flat.requires_grad=False
        boxes_features_all=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*MAX_N, D, K, K,
        
        boxes_features_all=boxes_features_all.reshape(B*T,MAX_N,-1)  #B*T,MAX_N, D*K*K
        
        # Embedding 
        boxes_features_all=self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
        boxes_features_all=F.relu(boxes_features_all)
        boxes_features_all=self.dropout_emb_1(boxes_features_all)
        
    
        actions_scores=[]
        activities_scores=[]
        bboxes_num_in=bboxes_num_in.reshape(B*T,)  #B*T,
        for bt in range(B*T):
        
            N=bboxes_num_in[bt]
            boxes_features=boxes_features_all[bt,:N,:].reshape(1,N,NFB)  #1,N,NFB
    
            boxes_states=boxes_features  

            NFS=NFB

            # Predict actions
            boxes_states_flat=boxes_states.reshape(-1,NFS)  #1*N, NFS
            actn_score=self.fc_actions(boxes_states_flat)  #1*N, actn_num
            actions_scores.append(actn_score)

            # Predict activities
            boxes_states_pooled,_=torch.max(boxes_states,dim=1)  #1, NFS
            boxes_states_pooled_flat=boxes_states_pooled.reshape(-1,NFS)  #1, NFS
            acty_score=self.fc_activities(boxes_states_pooled_flat)  #1, acty_num
            activities_scores.append(acty_score)

        actions_scores=torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        activities_scores=torch.cat(activities_scores,dim=0)   #B*T,acty_num
        
#         print(actions_scores.shape)
#         print(activities_scores.shape)
       
        return actions_scores, activities_scores


class Basenet_distant(nn.Module):
    """
    main module of base model for collective dataset
    """

    def __init__(self, cfg):
        super(Basenet_distant, self).__init__()
        self.cfg = cfg

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        #   self.backbone=MyVGG16(pretrained=True)

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)

        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.dropout_emb_1 = nn.Dropout(p=self.cfg.train_dropout_prob)
        #         self.nl_emb_1=nn.LayerNorm([NFB])

        self.fc_actions = nn.Linear(NFB, self.cfg.num_actions)
        self.fc_activities = nn.Linear(NFB, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def savemodel(self, filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'fc_emb_state_dict': self.fc_emb_1.state_dict(),
            'fc_actions_state_dict': self.fc_actions.state_dict(),
            'fc_activities_state_dict': self.fc_activities.state_dict()
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
        with autocast():
            images_in, boxes_in, bboxes_num_in = batch_data

            # read config parameters
            B = images_in.shape[0]
            T = images_in.shape[1]
            H, W = self.cfg.image_size
            OH, OW = self.cfg.out_size
            MAX_N = self.cfg.num_boxes
            NFB = self.cfg.num_features_boxes
            NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
            EPS = 1e-5

            D = self.cfg.emb_features
            K = self.cfg.crop_size[0]

            # Reshape the input data
            images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
            boxes_in = boxes_in.reshape(B*T, MAX_N, 4)

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

            features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

            boxes_in_flat = torch.reshape(boxes_in, (B *T* MAX_N, 4))  # B*T*MAX_N, 4

            boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B*T)]
            boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
            boxes_idx_flat = torch.reshape(boxes_idx, (B*T* MAX_N,))  # B*T*MAX_N,

            # RoI Align
            boxes_in_flat.requires_grad = False
            boxes_idx_flat.requires_grad = False
            boxes_features_all = self.roi_align(features_multiscale,
                                                boxes_in_flat,
                                                boxes_idx_flat)  # B*T*MAX_N, D, K, K,

            boxes_features_all = boxes_features_all.reshape(B*T, MAX_N, D*K*K)  # B*T,MAX_N, D*K*K
            # Embedding
            boxes_features_all = self.fc_emb_1(boxes_features_all)  # B*T,MAX_N, NFB
            boxes_features_all = F.relu(boxes_features_all)
            boxes_features_all = self.dropout_emb_1(boxes_features_all)

            # actions_scores = []
            activities_scores = []
            bboxes_num_in = bboxes_num_in.reshape(B*T)
            # for b in range(B*T):
            #     N = bboxes_num_in[b]
            #     boxes_features = boxes_features_all[b, :N, :].reshape(1, N, NFB)  # 1,N,NFB

            #     boxes_states = boxes_features

            NFS = NFB

            # Predict actions
            boxes_states_flat = boxes_features_all.reshape(-1, NFS)  # B*T*MAX_N, NFS
            actions_scores = self.fc_actions(boxes_states_flat)  # B*T*MAX_N,actn_num

            # Predict activities
            who_interactor=1-actions_scores.softmax(dim=1)[:,0].reshape(B*T,MAX_N,1) # B*T,MAX_N,1, Since action 0 is the background action.
            boxes_states_pooled, _ = torch.max(boxes_features_all*who_interactor, dim=1)  # B*T,NFS, I want to predict the interaction based on the two interactors' actions.
            activities_scores = self.fc_activities(boxes_states_pooled)  # B*T,acty_num

            #   print(actions_scores.shape)
            #   print(activities_scores.shape)

            return actions_scores, activities_scores

import os,sys
import numpy  as np 
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
class Basenet_social(nn.Module):
    """
    main module of base model for collective dataset
    """

    def __init__(self, cfg,edge_types = 6+1,time_step =1):
        super(Basenet_social, self).__init__()
        self.cfg = cfg

        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        #   self.backbone=MyVGG16(pretrained=True)

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # self.roi_align = RoIAlign(*self.cfg.crop_size)

        state_dim=512 #D*F*K*K
        self.time_step = time_step
        self.state_dim = state_dim
        self.edge_types = edge_types
        self.edge_fcs = nn.ModuleList()
        self.compress_fc=nn.Linear(D*K*K,state_dim)

        for i in range(self.edge_types):
            # incoming and outgoing edge embedding
            edge_fc = nn.Linear(self.state_dim, self.state_dim)
            self.edge_fcs.append(edge_fc)
            


        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid())
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid() )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Tanh() )
        
        self.edge_attens = nn.ModuleList()

        for i in range(self.edge_types):
            edge_attention =  nn.Sequential(
                #nn.Dropout(),
                #nn.Linear(state_dim * 2, 4096),
                #nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(state_dim * 2, 1),
                nn.Sigmoid(),
                )
            self.edge_attens.append(edge_attention)


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
        with autocast():
            images_in, boxes_in, bboxes_num_in = batch_data

            # read config parameters
            B = images_in.shape[0]
            T = images_in.shape[1]
            H, W = self.cfg.image_size
            OH, OW = self.cfg.out_size
            MAX_N = self.cfg.num_boxes
            NFB = self.cfg.num_features_boxes
            NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn
            EPS = 1e-5

            D = self.cfg.emb_features
            K = self.cfg.crop_size[0]

            # Reshape the input data
            images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W
            boxes_in = boxes_in.reshape(B*T, MAX_N, 4)

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

            features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW

            boxes_in_flat = torch.reshape(boxes_in, (B *T* MAX_N, 4))  # B*T*MAX_N, 4
            boxes_in_flat=torch.split(boxes_in_flat,MAX_N,dim=0)


            # RoI Align
            boxes_features_all = roi_align_t(features_multiscale,
                                                    boxes_in_flat,
                                                    self.cfg.crop_size)  # B*T*MAX_N, D, K, K,
                
            boxes_features_all = boxes_features_all.reshape(B*T, MAX_N, D*K*K)  # B*T,MAX_N, D*K*K
            pose_features= self.compress_fc(boxes_features_all) # B*T,MAX_N,state_dim

            node_num = MAX_N

            # full_mask=torch.ones((B,MAX_N,MAX_N,1))
            # full_mask = full_mask.view(-1,node_num,node_num,1).repeat(1,1,1,self.edge_types).float()
            # full_mask = full_mask.detach()
            inputs=pose_features
            prop_state = inputs 

            all_scores = []

            for t in range(1 + self.time_step):
                
                message_states = []
                
                for i in range(self.edge_types):
                    message_states.append(self.edge_fcs[i](prop_state))

                message_states_torch = torch.cat(message_states,dim=2).contiguous()
                message_states_torch = message_states_torch.view(-1,node_num * self.edge_types,self.state_dim)

                relation_scores = []

                for i in range(self.edge_types):
                    relation_feature = message_states[i]
                    feature_row_large = relation_feature.contiguous().view(-1,node_num,1,self.state_dim).repeat(1,1,node_num,1)
                    feature_col_large = relation_feature.contiguous().view(-1,1,node_num,self.state_dim).repeat(1,node_num,1,1)
                    feature_large = torch.cat((feature_row_large,feature_col_large),3)
                    relation_score = self.edge_attens[i](feature_large) # B,MAX_N,MAX_N,1
                    relation_scores.append(relation_score)

                graph_scores = torch.cat(relation_scores,dim=3).contiguous() #B,MAX_N,MAX_N,7
                graph_scores = graph_scores# * full_mask
                all_scores.append(graph_scores)
                #graph_scores = graph_scores.detach()
                graph_scores = graph_scores.view(-1,node_num,node_num * self.edge_types)
                merged_message = torch.bmm(graph_scores, message_states_torch) # B,MAX_N,state_dim 


                a = torch.cat((merged_message,prop_state),2)

                r = self.reset_gate(a)
                z = self.update_gate(a)
                joined_input = torch.cat((merged_message, r * prop_state), 2)
                h_hat = self.tansform(joined_input)
                prop_state = (1 - z) * prop_state + z * h_hat
            # Below, let's translate their results into our output!
            box_states=all_scores[-1]
            action_score=((1-box_states[...,0,None])*box_states).sum(dim=2).reshape(B,T,MAX_N,self.edge_types) #B,T,MAX_N,7
            actor_score=(1-box_states[...,0]).reshape(B,T,MAX_N,MAX_N)# Here, T=1.
            acty_score=torch.softmax((action_score*actor_score.sum(dim=-1)[...,None]).sum(dim=2)[:,:,1:],dim=-1) # B,T,6
            return action_score,acty_score, actor_score