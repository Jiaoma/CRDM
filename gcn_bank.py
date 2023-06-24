from gcn_model import *

class GCN_BANK_Module(nn.Module):
    '''
    Function description:
    1. Input appearance, xy, dxdy of each subject.
    2. Build node and edge.
    3. Init bank.
    4. Save identity of each node as key, input identity into GRU, save outputs of GRU as value.
    5. For the following nodes, firstly associate with existed key. If failed create new key. Else, replace value and
    key in the dict. 
    6. Return current outputs of GRU for calculating loss and update parameters. 
    7. Notice that values in bank are deteach from auto-graph of PyTorch.
    '''
    def __init__(self, cfg,number_feature_relation,num_features_gcn,num_graph):
        super(GCN_BANK_Module, self).__init__()

        self.cfg = cfg

        self.num_features_relation=number_feature_relation
        num_graph=1
        self.num_graph = num_graph

        input_channels=cfg.num_features_pose+4 # Before GRU

        self.num_features_gcn=num_features_gcn
        self.num_feature_pose=self.cfg.num_features_pose

        self.adj_motion_multi_fc=torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    nn.Linear(4,64),
                    nn.Linear(64,1)
                ) 
                for i in range(num_graph)]
        )
        self.adj_app_fc=torch.nn.ModuleList(
            [nn.Linear(cfg.num_features_pose,1) for i in range(num_graph)]
        )

        self.fc_gcn_list = torch.nn.ModuleList(
            [nn.Linear(input_channels, num_features_gcn, bias=False) for i in range(num_graph)])

        self.nl_gcn_list = torch.nn.ModuleList(
                [nn.LayerNorm([num_features_gcn]) for i in range(num_graph)])

    def forward(self, app_features, position,dxdy):
        """
        app_features  [B,T,N,number_feature_pose]
        position [B,T,N,2] # Instead, we need B,T,N,2 now.
        dxdy [B,T,N,2]

        return node features, relations
        Outside:
        sum up relations,
        """
        # GCN graph modeling
        # Prepare boxes similarity relation
        B, T, N, number_feature_pose = app_features.shape
        boxes_exist=position.sum(dim=3)
        boxes_exist=boxes_exist>0 # B,T,N
        device=app_features.device
        motion_features=torch.cat([position,dxdy],dim=3) # B,T,N,2+2
        # build graph
        # init edge
        edge_adj_motion_matrix=torch.sigmoid(self.adj_motion_multi_fc((motion_features[:,:,:,None,:]-motion_features[:,:,None,:,:])).reshape(-1,4)) # B*T*N*N (4->C1->1)
        edge_adj_app_matrix=torch.sigmoid(self.adj_app_fc((torch.mul(app_features[:,:,:,None,:],app_features[:,:,None,:,:]))).reshape(-1,number_feature_pose)) # B*T*N*N (num_app->C2->1)
        edge_adj_matrix=torch.softmax((edge_adj_app_matrix+edge_adj_motion_matrix).reshape(B,T,N,N),dim=3) #B,T,N,N, since in motion_matrix, the diagnoal items are near zero, we need to add themselves in aggregating step.
        # aggregate node information
        init_node_feature=torch.cat([app_features,motion_features],dim=3) # B,T,N,num_app+4
        # This is the key in bank too.
        A_node_info=torch.matmul(edge_adj_matrix,init_node_feature)+init_node_feature # B,T,N,num_app+4
        # Graph convolution
        one_graph_boxes_features = self.fc_gcn_list[0](A_node_info)  # B,T, N, number_feature_gcn_ONE
        one_graph_boxes_features = self.nl_gcn_list[0](one_graph_boxes_features)

        return one_graph_boxes_features, edge_adj_matrix