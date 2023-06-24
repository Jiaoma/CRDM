import os
from numpy.lib.arraysetops import isin

import torch
import torch.nn as nn
import torch.autograd

from torch.cuda.amp import autocast

from distant_utils import generate_actors_from_action

import units

class GPNN_CAD(torch.nn.Module):
    def __init__(self, cfg):
        super(GPNN_CAD, self).__init__()
        model_args = {
            'edge_feature_size':128,
            'message_size':128,
            'node_feature_size':64,
            'hoi_classes':7,
            'propagate_layers':1,
            'resize_feature_to_message_size':128,
            'subactivity_classes':7,
            'affordance_classes':7,
            'link_hidden_size':128,
            'link_hidden_layers':2
        }
        self.model_args=model_args

        self.cfg = cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        F = self.cfg.continuous_frames
        number_feature_relation, number_feature_gcn = self.cfg.num_features_relation, self.cfg.num_features_gcn
        MAX_N = self.cfg.num_boxes
        self.R2 = nn.Parameter(torch.tensor([_i for _i in range(1,MAX_N+1)],dtype=torch.float32), requires_grad=False)[None,:] # 1,2,3,4,...

        self.compress_fc=nn.Linear(D*F*K*K,model_args['node_feature_size'])
        self.compress=nn.Linear(2*model_args['edge_feature_size'],model_args['edge_feature_size'])

        self.link_fun = units.LinkFunction('GraphConvLSTM', model_args)
        self.message_fun = units.MessageFunction('linear_concat', model_args)

        self.update_funs = torch.nn.ModuleList([])
        self.update_funs.append(units.UpdateFunction('gru', model_args))
        self.update_funs.append(units.UpdateFunction('gru', model_args))

        self.subactivity_classes = model_args['subactivity_classes']
        self.affordance_classes = model_args['affordance_classes']
        self.readout_funs = torch.nn.ModuleList([])
        self.readout_funs.append(units.ReadoutFunction('fc_soft_max', {'readout_input_size': model_args['node_feature_size'], 'output_classes': self.subactivity_classes}))
        self.readout_funs.append(units.ReadoutFunction('fc_soft_max', {'readout_input_size': model_args['node_feature_size'], 'output_classes': self.affordance_classes}))

        self.propagate_layers = model_args['propagate_layers']
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
            elif isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight)

        # self._load_link_fun(model_args)
    @autocast()
    def forward(self, batch_data):
        with autocast():
            pose_features, boxes_in, bboxes_num_in, boxes_dhdw, actor_label = batch_data

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
                pose_features = pose_features.reshape(B,T,MAX_N, D*F*K*K)  # B,T,MAX_N, D,F,K,K
                
            #     boxes_center = (boxes_in[:, :, :, :2,:] + boxes_in[:, :, :, 2:, :]) / 2 # B,T,N,2,F

            #     # boxes_center=boxes_center.transpose(3,4)#B,T,MAX_N,2
            #     boxes_dhdw=boxes_dhdw[...]#B,T,MAX_N,2,F
            
            #     boxes_exist=boxes_center[...,0].sum(dim=3)
            #     boxes_exist=boxes_exist>0 # B,T,N
            #     boxes_exist_int=boxes_exist.float()
            #     # Below is the old version, it's totally wrong
            #     boxes_exist_matrix=boxes_exist[:,:,:,None,None]*(boxes_exist[:,:,None,:,None]) # B,T,MAX_N,MAX_N,1
            #     boxes_exist_matrix_line=boxes_exist[:,:,:,None,None].expand(B,T,MAX_N,MAX_N,1) # B,T,MAX_N,MAX_N,1
            #     conflicts=torch.matmul(boxes_exist_int.transpose(1,2),boxes_exist_int)
            #     conflicts=conflicts==0 # B,N,N     main dim is the second N
            #     # bboxes_num_in=torch.zeros((B,),dtype=torch.int)
            #     temp=boxes_exist.sum(dim=1) #B,N
            #     temp=(temp>0).float()
                
            #     temp_index=temp*self.R2
            #     bboxes_num_in=(torch.argmax(temp_index,dim=1)+1).tolist() # B
            # all_features= pose_features.reshape(B,T,MAX_N,D*F*K*K)
            
            # # Calculate mean node features
            # mean_node_features=torch.sum(all_features*boxes_exist[...,None],dim=1,keepdim=True)/(torch.sum(boxes_exist[...,None].float(),dim=1,keepdim=True)+1e-12) # B,MAX_N,NFG
            # # f=lambda x,y: torch.abs(x-y).sum(dim=-1)
            # mask=~boxes_exist*temp[:,None,:]
            # select=torch.nonzero(mask).tolist()
            # for (b,t,box) in select:
            #     if box>=bboxes_num_in[b]:
            #         continue
            #     if mask[b,t,box]:
            #         other_node_index=conflicts[b,:,box]&boxes_exist[b,t]
            #         have_other=other_node_index.sum()
            #         if have_other:
            #             # local_node_graph_features=node_graph_features[b,t]
            #             other_node_features=all_features[b,t,other_node_index] # X, NFG 
            #             min_distant_index=torch.argmin(
            #                 torch.abs(mean_node_features[b,:,box,:]-other_node_features).sum(dim=-1)) # 1
            #             all_features[b,t,box].add_(other_node_features[min_distant_index]) # Copy here.
            #         else:
            #             all_features[b,t,box].add_(mean_node_features[b,0,box])
            
            pose_features=self.compress_fc(pose_features)
            
            # del all_features
            # del conflicts
            # del mask
            # del temp
            # del mean_node_features
            # del select

            node_features=pose_features
            temp_feature=node_features[:,:,:,None,:].expand((B,T,MAX_N,MAX_N,self.model_args['node_feature_size']))

            edge_features=torch.cat([temp_feature,temp_feature.transpose(2,3)],dim=-1)
            node_features=node_features.reshape(B*T,MAX_N,-1).transpose(1,2)
            edge_features=edge_features.reshape(B*T,MAX_N,MAX_N,-1).permute(0,3,1,2)
            # pred_adj_mat = self.link_fun(edge_features)
            # pred_adj_mat = torch.autograd.Variable(torch.ones(adj_mat.size())).cuda()  # Test constant graph
            pred_node_labels = torch.autograd.Variable(torch.zeros((B*T,MAX_N,7)))
            args=self.model_args
            pred_node_labels = pred_node_labels.cuda()
            hidden_node_states = [node_features.clone() for passing_round in range(self.propagate_layers+1)]
            hidden_edge_states = [edge_features.clone() for passing_round in range(self.propagate_layers+1)]

            # Belief propagation
            for passing_round in range(self.propagate_layers):
                pred_adj_mat = self.link_fun(hidden_edge_states[passing_round])
                # if passing_round == 0:
                #     pred_adj_mat = torch.autograd.Variable(torch.ones(adj_mat.size())).cuda()  # Test constant graph
                #     pred_adj_mat = self.link_fun(hidden_edge_states[passing_round]) # Without iterative parsing

                # Loop through nodes
                for i_node in range(node_features.size()[2]):
                    h_v = hidden_node_states[passing_round][:, :, i_node]
                    h_w = hidden_node_states[passing_round]
                    e_vw = edge_features[:, :, i_node, :]
                    m_v = self.message_fun(h_v, h_w, e_vw, args)

                    # Sum up messages from different nodes according to weights
                    m_v = pred_adj_mat[:, i_node, :].unsqueeze(1).expand_as(m_v) * m_v
                    hidden_edge_states[passing_round+1][:, :, :, i_node] = m_v
                    m_v = torch.sum(m_v, 2)
                    if i_node == 0:
                        h_v = self.update_funs[0](h_v[None].contiguous(), m_v[None])
                    else:
                        h_v = self.update_funs[1](h_v[None].contiguous(), m_v[None])

                    # Readout at the final round of message passing
                    if passing_round == self.propagate_layers-1:
                        if i_node == 0:
                            pred_node_labels[:, i_node, :self.subactivity_classes] = self.readout_funs[0](h_v.squeeze(0))
                        else:
                            pred_node_labels[:, i_node, :] = self.readout_funs[1](h_v.squeeze(0))

            '''
            pred_adj_mat: B x T x MAX_N x MAX_N
            pred_node_labels: B x T x MAX_N x act_num
            '''
            actor_score=pred_adj_mat.reshape(B,T,MAX_N*MAX_N)
            actor_score=torch.softmax(actor_score,dim=-1).reshape(B,T,MAX_N,MAX_N)
            actor_score=actor_score-torch.diag_embed(torch.diagonal(actor_score,dim1=-2,dim2=-1),dim1=-2,dim2=-1)
            action_score=pred_node_labels.reshape(B,T,MAX_N,7)
            # actor_score=1-action_score[...,0]
            # actor_score=generate_actors_from_action(actor_score.reshape(B*T,MAX_N)).reshape(B,T,MAX_N,MAX_N)
            acty_score=(action_score*actor_score.sum(dim=-1,keepdim=True))[...,1:].sum(dim=2) # B x T x acty_num
            return action_score,acty_score, actor_score

    def _load_link_fun(self, model_args):
        if not os.path.exists(model_args['model_path']):
            os.makedirs(model_args['model_path'])
        best_model_file = os.path.join(model_args['model_path'], '..', 'graph', 'model_best.pth')
        if os.path.isfile(best_model_file):
            checkpoint = torch.load(best_model_file)
            self.link_fun.load_state_dict(checkpoint['state_dict'])

    def _dump_link_fun(self, model_args):
        if not os.path.exists(model_args['model_path']):
            os.makedirs(model_args['model_path'])
        if not os.path.exists(os.path.join(model_args['model_path'], '..', 'graph')):
            os.makedirs(os.path.join(model_args['model_path'], '..', 'graph'))
        best_model_file = os.path.join(model_args['model_path'], '..', 'graph', 'model_best.pth')
        torch.save({'state_dict': self.link_fun.state_dict()}, best_model_file)