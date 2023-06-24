import os,sys
import numpy  as np 
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class GRRN(nn.Module):
    def __init__(self,cfg, edge_types = 6+1,time_step = 5):
        super(GRRN, self).__init__()
        self.cfg=cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        F = self.cfg.continuous_frames
        state_dim=1024 #D*F*K*K
        self.time_step = time_step
        self.state_dim = state_dim
        self.edge_types = edge_types
        self.edge_fcs = nn.ModuleList()
        self.compress_fc=nn.Linear(D*F*K*K,state_dim)

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

        self._initialization()


    # inputs with feature dim [batch, node_num, hidden_state_dim]
    # A with feature dim [batch, node_num, node_num]
    # reture output with feature dim [batch, node_num, output_dim]
    def forward(self,batch_data):
        pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data

        # read config parameters
        B = boxes_in.shape[0]
        T = boxes_in.shape[1]
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        F = self.cfg.continuous_frames

        MAX_N = self.cfg.num_boxes

        if self.training:
            B = B * 3
            T = T // 3
            pose_features.resize_((B, T) + pose_features.shape[2:])
            boxes_in.resize_((B, T) + boxes_in.shape[2:])
            bboxes_num_in.resize_((B, T))
            boxes_dhdw.resize_((B, T) + boxes_dhdw.shape[2:])


        pose_features = pose_features.reshape(B*T,MAX_N, D*F*K*K)  # B,T,MAX_N, D,F,K,K
        pose_features= self.compress_fc(pose_features) # B*T,MAX_N,state_dim

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

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)


class GRRN_GRU(nn.Module):
    def __init__(self,cfg, edge_types = 6+1,time_step = 5):
        super(GRRN_GRU, self).__init__()
        self.cfg=cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        F = self.cfg.continuous_frames
        state_dim=1024 #D*F*K*K
        self.time_step = time_step
        self.state_dim = state_dim
        self.edge_types = edge_types
        self.edge_fcs = nn.ModuleList()
        self.compress_fc=nn.Linear(D*F*K*K,state_dim)
        self.compress=nn.Linear(2*state_dim,state_dim)
        

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
        
        self.gru=nn.GRU(state_dim,state_dim,num_layers=1,bidirectional=False)

        self.edge_attens = nn.ModuleList()

        for i in range(self.edge_types):
            edge_attention =  nn.Sequential(
                #nn.Dropout(),
                #nn.Linear(state_dim * 2, 4096),
                #nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(state_dim, 1),
                nn.Sigmoid(),
                )
            self.edge_attens.append(edge_attention)

        self._initialization()


    # inputs with feature dim [batch, node_num, hidden_state_dim]
    # A with feature dim [batch, node_num, node_num]
    # reture output with feature dim [batch, node_num, output_dim]
    def forward(self,batch_data):
        pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data

        # read config parameters
        B = boxes_in.shape[0]
        T = boxes_in.shape[1]
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        F = self.cfg.continuous_frames
        device=pose_features.device

        MAX_N = self.cfg.num_boxes

        if self.training:
            B = B * 3
            T = T // 3
            pose_features.resize_((B, T) + pose_features.shape[2:])
            boxes_in.resize_((B, T) + boxes_in.shape[2:])
            bboxes_num_in.resize_((B, T))
            boxes_dhdw.resize_((B, T) + boxes_dhdw.shape[2:])


        pose_features = pose_features.reshape(B*T,MAX_N, D*F*K*K)  # B,T,MAX_N, D,F,K,K
        pose_features= self.compress_fc(pose_features) # B*T,MAX_N,state_dim

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
                feature_large = torch.cat((feature_row_large,feature_col_large),3).reshape(B,T,MAX_N,MAX_N,2*self.state_dim).permute(1,0,2,3,4).reshape(T,B*MAX_N*MAX_N,2*self.state_dim)
                feature_large=self.compress(feature_large)
                ####### insert GRU
                h1 = torch.empty((1, B*MAX_N*MAX_N, self.state_dim), device=device)
                torch.nn.init.xavier_normal_(h1)

                self.gru.flatten_parameters()
                gru_outputs,_=self.gru(feature_large,h1)

                gru_outputs=gru_outputs.reshape(T, B,MAX_N,MAX_N, self.state_dim).transpose(0,1).reshape(B*T,MAX_N,MAX_N,-1)
                #######
                relation_score = self.edge_attens[i](gru_outputs) # B*T,MAX_N,MAX_N,1
                relation_scores.append(relation_score)

            graph_scores = torch.cat(relation_scores,dim=3).contiguous() #B*T,MAX_N,MAX_N,7
            graph_scores = graph_scores# * full_mask
            all_scores.append(graph_scores)
            #graph_scores = graph_scores.detach()
            graph_scores = graph_scores.view(-1,node_num,node_num * self.edge_types)
            merged_message = torch.bmm(graph_scores, message_states_torch) # B*T,MAX_N,state_dim 


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

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)