from numpy.core.fromnumeric import shape
import torch
import torch.optim as optim

import time
import random
import os
import sys
import pickle

from config import *
from volleyball import *
from collective import *
from dataset import *
from gcn_model import *
from base_model import *
from utils import *
from evaluate import evaluateDistant
from distant_utils import saveDict,generate_actors


from torch.cuda.amp import autocast,GradScaler

note_net_map={
    'chance':OneGroupV5Net,#get
    'Distant_stage2_OneGroupV5Net':OneGroupV5Net,#get
    'Distant_stage2_OneGroupV5Net_K5_F3':OneGroupV5Net,#get
    'Distant_stage2_OneGroupV5Net_K5_F5':OneGroupV5Net,#get
    'Distant_stage2_OneGroupV5Net_K7_F3':OneGroupV5Net,#get
    'Distant_stage2_OneGroupV5Net_woaction':OneGroupV5Net, 
    'Distant_stage2_OneGroupV5Net_woMOTrepair':OneGroupV5Net_woMOTrepair,  #ok
    'Distant_stage2_OneGroupV5Net_wotraj':OneGroupV5Net_wotraj, #ok,rerun
    'Distant_stage2_OneGroupV5Net_woapp':OneGroupV5Net_woapp, #running, need record, rerun
    'Distant_stage2_OneGroupV5Net_TSN':OneGroupV5Net_TSN, # ok
    'Distant_stage2_OneGroupV5Net_actor':OneGroupV5Net_actor, #ok
    'Distant_stage2_OneGroupV5Net_wolong':OneGroupV5Net_wolong, #ok
    'Distant_stage2_OneGroupV5Net_wotriplet':OneGroupV5Net, #ok,
    'Distant_stage2_OneGroupV5Net_GCN':OneGroupV5Net_GCN
}

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train_net(cfg,training_set=None,validation_set=None):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    # vis=Visdom_E()
    # vis.set('tsne',env='hit')

    # Reading dataset
    if (training_set is None) or (validation_set is None):
        training_set, validation_set = return_dataset(cfg)

    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': cfg.num_workers,
        'pin_memory': True,
        'drop_last':True
    }

    params_test={
        'batch_size': cfg.test_batch_size,
        'shuffle': False,
        'num_workers': cfg.num_workers,
        'pin_memory': True
    }
    training_loader = data.DataLoader(training_set, **params)

    params['batch_size'] = cfg.test_batch_size
    validation_loader = data.DataLoader(validation_set, **params_test)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.backends.cudnn.deterministic = True

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    DistantNet = note_net_map[cfg.exp_note.split('+')[0]]
    # Build model and optimizer
    basenet_list = {'Distant': Basenet_distant}
    gcnnet_list = {'Distant': DistantNet}

    if cfg.training_stage == 1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(cfg)
        # model.loadmodel(cfg.stage1_model_path)
    elif cfg.training_stage == 2:
        GCNnet = gcnnet_list[cfg.dataset_name]
        model = GCNnet(cfg)
        # Load backbone
        # model.loadmodel(cfg.stage1_model_path)
        if cfg.stage2_model_path!='':
            model.loadmodel_normal(cfg.stage2_model_path)
    else:
        assert(False)

    if cfg.use_multi_gpu:
        model = nn.DataParallel(model)

    model = model.to(device=device)
    scaler = GradScaler()

    model.train()
    # model.apply(set_bn_eval)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)

    if cfg.training_stage==1:
        train_list = {'Distant': train_distant_stage1}
    else:
        train_list = {'Distant': train_distant_stage2}
    test_list = {'Distant': test_distant}
    train = train_list[cfg.dataset_name]
    test = test_list[cfg.dataset_name]

    if cfg.test_before_train:
        test_info = test(validation_loader, model, device, 0, cfg,scaler)
        print(test_info)

    # Training iteration
    best_result = {'epoch': 0, 'activities_acc': 0}
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):

        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
        if epoch==10:
            print('debug')
        # One epoch of forward and backward
        train_info = train(training_loader, model,
                           device, optimizer, epoch, cfg,scaler)
        show_epoch_info('Train', cfg.log_path, train_info)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info = test(validation_loader, model, device, epoch, cfg,scaler)
            show_epoch_info('Test', cfg.log_path, test_info)

            if test_info['activities_acc'] >= best_result['activities_acc']:
                best_result = test_info
            print_log(cfg.log_path,
                      'Best group activity accuracy: %.2f%% at epoch #%d.' % (best_result['activities_acc'], best_result['epoch']))

            # Save model STAGE1_MODEL
            if cfg.training_stage == 2:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                filepath = cfg.result_path + \
                    '/stage%d_epoch%d_%.2f%%.pth' % (
                        cfg.training_stage, epoch, test_info['activities_acc'])
                torch.save(state, filepath)
                print('model saved to:', filepath)
            elif cfg.training_stage == 1:
                for m in model.modules():
                    if isinstance(m, Basenet):
                        filepath = cfg.result_path + \
                            '/stage%d_epoch%d_%.2f%%.pth' % (
                                cfg.training_stage, epoch, test_info['activities_acc'])
                        m.savemodel(filepath)
#                         print('model saved to:',filepath)
            else:
                assert False


def test_net(cfg,training_set=None,validation_set=None):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parametersx
    cfg.init_config()
    show_config(cfg)
    # vis=Visdom_E()
    # plot_actor=result_real_time_show(cfg,'actor',vis)
    # plot_others=result_real_time_show(cfg,'others',vis)

    # Reading dataset
    if (training_set is None) or (validation_set is None):
        training_set, validation_set = return_dataset(cfg)


    params = {
        'batch_size': cfg.test_batch_size,
        'shuffle': True,
        'num_workers': cfg.num_workers
    }

    params['batch_size'] = cfg.test_batch_size
    validation_loader = data.DataLoader(validation_set, **params)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.backends.cudnn.deterministic = True

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    DistantNet = note_net_map[cfg.exp_note.split('+')[0]]


    # Build model and optimizer
    basenet_list = {'Distant': Basenet_distant}
    gcnnet_list = {'Distant': DistantNet}

    if cfg.training_stage == 1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(cfg)
        model.loadmodel(cfg.stage1_model_path)
    elif cfg.training_stage == 2:
        GCNnet = gcnnet_list[cfg.dataset_name]
        model = GCNnet(cfg)
        # Load backbone
        # model.loadmodel(cfg.stage1_model_path)
        if cfg.exp_note!='chance' and not hasattr(cfg,'save_result_path'):
            model.loadmodel_normal(cfg.stage2_model_path)
    else:
        assert (False)

    if cfg.use_multi_gpu:
        model = nn.DataParallel(model)

    model = model.to(device=device)

    model.train()
    scaler = GradScaler()
    # model.apply(set_bn_eval)

    test_list = {'Distant': test_distant}
    test = test_list[cfg.dataset_name]

    # test_info = test(validation_loader, model, device, 0, cfg,scaler,plot_actor,plot_others)
    test_info = test(validation_loader, model, device, 0, cfg,scaler)
    print_log(cfg.log_path,test_info)



def train_distant(data_loader, model, device, optimizer, epoch, cfg):
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)

        # prepare batch data
        batch_data = [b.to(device=device) for b in batch_data]
        batch_size = batch_data[0].shape[0]
        num_frames = batch_data[0].shape[1]

        # forward
        if cfg.training_stage==1:
            actions_scores, activities_scores = model(
                (batch_data[0], batch_data[1], batch_data[4]))
        else:
            actions_scores, activities_scores = model(
                (batch_data[0], batch_data[1], batch_data[4],batch_data[5],batch_data[6]))

        actions_in = batch_data[2].reshape(
            (batch_size, num_frames, cfg.num_boxes))[:,0,:] # I only need the sequence-level acc.
        activities_in = batch_data[3].reshape((batch_size, num_frames))[:,0]
        bboxes_num = batch_data[4].reshape(batch_size, num_frames)[:,0]

        actions_in_nopad = []
        for b in range(batch_size):
            N = bboxes_num[b]
            actions_in_nopad.append(actions_in[b][:N])
        actions_in = torch.cat(actions_in_nopad, dim=0).reshape(-1, )  # ALL_N,

        # activities_in = activities_in[:].reshape(batch_size, )

        # Predict actions
        actions_loss = F.cross_entropy(actions_scores, actions_in, weight=None)
        actions_labels = torch.argmax(actions_scores, dim=1)  # B*N,
        actions_correct = torch.sum(
            torch.eq(actions_labels.int(), actions_in.int()).float())

        # Predict activities
        activities_loss = F.cross_entropy(activities_scores, activities_in)
        activities_labels = torch.argmax(activities_scores, dim=1)  # B*T,
        activities_correct = torch.sum(
            torch.eq(activities_labels.int(), activities_in.int()).float())

        # Get accuracy
        actions_accuracy = actions_correct.item() / actions_scores.shape[0]
        activities_accuracy = activities_correct.item() / \
            activities_scores.shape[0]

        actions_meter.update(actions_accuracy, actions_scores.shape[0])
        activities_meter.update(activities_accuracy,
                                activities_scores.shape[0])

        # Total loss
        total_loss = activities_loss + cfg.actions_loss_weight * actions_loss
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        'actions_acc': actions_meter.avg * 100
    }

    return train_info

# @profile
def test_distant(data_loader, model, device, epoch, cfg,scaler):
    model.eval()
    ablation=cfg.ablation

    epoch_timer = Timer()

    if not hasattr(cfg,'save_result_path'):
        # Class correct num
        ACTIVITIES = ['ch','co','ga','hl','pp','tc']

        ACTIONS=['NA','ch','co','ga','hl','pp','tc']

        activities_result_seq_collection={'prediction':{},'gt':{}}
        actors_result_seq_collection={'prediction':{},'gt':{}}
        actions_results_seq_collection={'prediction':{},'gt':{}}
        detection_failed_num_dict={}

        # activities_each_acc_dict=OrderedDict()
        # activities_each_num_dict=OrderedDict()

        # actions_each_acc_dict=OrderedDict()
        # actions_each_num_dict=OrderedDict()
        MAX_N=cfg.num_boxes

        # for i in range(len(ACTIVITIES)):
        #     activities_each_acc_dict[ACTIVITIES[i]] = 0
        #     activities_each_num_dict[ACTIVITIES[i]] = 0

        # for i in range(len(ACTIONS)):
        #     actions_each_acc_dict[ACTIONS[i]] = 0
        #     actions_each_num_dict[ACTIONS[i]] = 0

        with torch.no_grad():
            for batch_data in data_loader:
                # prepare batch data
                batch_data = [b.to(device=device) for b in batch_data]
                batch_size = batch_data[0].shape[0]
                num_frames = batch_data[0].shape[1]
                seq_ids=batch_data[6].long()

                actions_in=batch_data[2].reshape(batch_size,num_frames,cfg.num_boxes) # B,T,MAX_N
                actors_in=batch_data[7].reshape(batch_size,num_frames,cfg.num_boxes,cfg.num_boxes) #B,T,MAX_N,MAX_N
                # actors_in_ = actions_in.clamp_max(1)
                # actors_in=torch.diag_embed(actors_in_,dim1=-2,dim2=-1)
                activities_in = batch_data[3].reshape((batch_size, num_frames)) # B,T
                '''
                bboxes_num: (B,) Here, the number of bboxes in each seq is pre-calculated, they are different among different seqs but keep the same during one seq. If in one seq, the number of bbox is 
                changed, the changed one will be replaced by zero vector.
                '''
                bboxes_num = batch_data[4].reshape(batch_size, num_frames)[:,0] 
                # forward
                if cfg.exp_note == 'chance':
                    activities_scores=torch.randn(batch_size*num_frames,len(ACTIVITIES))
                    actor_scores=torch.softmax(torch.randn(batch_size,num_frames,MAX_N*MAX_N),dim=-1).reshape(batch_size,num_frames,MAX_N,MAX_N)
                    actions_scores=torch.randn(batch_size*num_frames,MAX_N,cfg.num_actions)
                else:
                    if cfg.training_stage==1:
                        with autocast():
                            actions_scores, activities_scores = model(
                            (batch_data[0], batch_data[1], batch_data[4]))
                        actors_ = torch.argmax(actions_scores,dim=1).clamp(0,1).reshape(batch_size,num_frames,MAX_N)
                        actor_scores=generate_actors(actors_) # B,T,N,N
                    else:
                        # actions_scores,activities_scores, actor_scores = model(
                        #             (batch_data[0], batch_data[1], batch_data[4], batch_data[5],actors_in),ablation,plot)
                        actions_scores,activities_scores, actor_scores = model(
                                    (batch_data[0], batch_data[1], batch_data[4], batch_data[5],actors_in),ablation)
                        # if ablation:
                        #     with autocast():
                        #         actions_scores,activities_scores, actor_scores = model(
                        #             (batch_data[0], batch_data[1], batch_data[4], batch_data[5],actors_in),ablation)
                        # else:
                        #     with autocast():
                        #         actions_scores,activities_scores, actor_scores = model(
                        #         (batch_data[0], batch_data[1], batch_data[4], batch_data[5]))
                        # actions_scores=actions_in
                # Note: In Version 2, the actor_scores are adjacent matrix in shape of (B,T,N,N).
                # I don't change the old name.
                actions_in=actions_in.cpu()
                activities_in=activities_in.cpu()
                actors_in=actors_in.cpu()
                actions_scores=actions_scores.cpu()
                activities_scores=activities_scores.cpu()
                actor_scores=actor_scores.cpu()
                detection_failed_num=batch_data[8].int().cpu()
                # Predict actions
                if ablation:
                    actor_scores = actors_in.float()
                else:
                    actor_scores=actor_scores.reshape(batch_size,num_frames,MAX_N,MAX_N)
                # plot2.input_image_gt(batch_data[0][0,:,0,...].detach().cpu(),batch_data[1][0,:,:,:,0].detach().cpu(),
                # activities_in[0],actors_in[0].sum(dim=-1),actions_in[0],activities_scores[0].detach().cpu(),actor_scores[0].sum(dim=-1).detach().cpu(),actions_scores[0].detach().cpu()
                # )
                # time.sleep(1)
                # actor_scores=1-torch.diagonal(actor_scores,dim1=-2,dim2=-1)
                # temp_actors=torch.zeros_like(actors_in) # B,T,MAX_N,MAX_N
                # _,actors_pred_index=torch.max(actor_scores,dim=3)
                # temp_actors.scatter_(3,actors_pred_index[...,None],1)
                # actor_scores=1- torch.diagonal(temp_actors,dim1=-2,dim2=-1) # B,T,MAX_N
                
                actors_labels_o=torch.sum(actor_scores,dim=3) # B,T,N
                # Below is the old solution, not good.
                # actors_labels/=(torch.max(actors_labels,dim=2,keepdim=True)[0]+1e-12) # B,T,N
                # actors_labels[actors_labels>0.5]=1
                # actors_labels[actors_labels<1]=0
                actors_labels_top2_value,actors_labels_top2=torch.topk(actors_labels_o,k=2,dim=-1)
                actors_labels=torch.zeros_like(actors_labels_o).scatter_(dim=-1,index=actors_labels_top2,src=torch.ones_like(actors_labels_o))
                actors_noise=(actors_labels_top2_value[...,0]-actors_labels_top2_value[...,1]).abs()/torch.max(actors_labels_top2_value,dim=-1)[0]
                actors_noise=(actors_noise>0.5)[...,None].expand_as(actors_labels) # The smaller one is much less than the bigger one, B,T,N
                actors_small=torch.min(actors_labels_top2_value,dim=-1)[0][...,None].expand_as(actors_labels) #B,T,N
                actors_labels[(actors_labels_o==actors_small)&actors_noise]=0 # Post-processing
                
                # actors_gt_labels=torch.argmax(actors_in,dim=3).reshape(-1)
                actors_in=actors_in.sum(dim=3) #.reshape(-1)
                
                # actor_scores=torch.sum(actor_scores,dim=3)
                # actors_in=1-torch.diagonal(actors_in,dim1=-2,dim2=-1)
                actions_scores=actions_scores.reshape(batch_size,num_frames,MAX_N,-1)
                # Predict activities
                activities_scores=activities_scores.reshape(batch_size,num_frames,-1)
                # Store actors, actions at each frame
                # Process and store activity at each seq
                for i_s in range(batch_size):
                    seq_id=seq_ids[i_s][0].item()
                    if seq_id not in activities_result_seq_collection['prediction'].keys():
                        activities_result_seq_collection['prediction'][seq_id] = []  # First collect, then average
                        actors_result_seq_collection['prediction'][seq_id] = []
                        actions_results_seq_collection['prediction'][seq_id]=[]
                        activities_result_seq_collection['gt'][seq_id] = activities_in[i_s,0] # A digit
                        actors_result_seq_collection['gt'][seq_id] = [] 
                        actions_results_seq_collection['gt'][seq_id]=[]
                        detection_failed_num_dict[seq_id]=[]
                    activities_result_seq_collection['prediction'][seq_id].append(activities_scores[i_s])
                    actors_result_seq_collection['prediction'][seq_id].append(actors_labels[i_s,:,:bboxes_num[i_s]].int()) # T, N
                    actions_results_seq_collection['prediction'][seq_id].append(torch.argmax(actions_scores[i_s,:,:bboxes_num[i_s],:],dim=2).int()) # T, N
                    # actions_results_seq_collection['prediction'][seq_id].append(actions_in[i_s,:,:bboxes_num[i_s]]) # T, N
                    actors_result_seq_collection['gt'][seq_id].append(actors_in[i_s,:,:bboxes_num[i_s]].int())
                    actions_results_seq_collection['gt'][seq_id].append(actions_in[i_s,:,:bboxes_num[i_s]])
                    # print(detection_failed_num)
                    # print(detection_failed_num.shape)
                    # exit(0)
                    detection_failed_num_dict[seq_id].append(detection_failed_num[i_s])# BUG!!! It cause the action score very low.
        print('data collection okay')
        for seq in activities_result_seq_collection['prediction'].keys():
            activities_result_seq_collection['prediction'][seq]=torch.argmax(torch.cat(activities_result_seq_collection['prediction'][seq],dim=0).mean(dim=0)).item() # 1
        # Get average
        final_result={'actions':actions_results_seq_collection,
                    'actors':actors_result_seq_collection,
                    'interactions':activities_result_seq_collection,
                    'detection_failed_num':detection_failed_num_dict}
        # for seq_id in activities_result_seq_collection.keys():
        #     ave_activities=torch.stack(activities_result_seq_collection[seq_id]) # -1, num_activities
        #     ave_activities=torch.mean(ave_activities,dim=0)
        #     ave_activities=torch.argmax(ave_activities,dim=0).item()
        #     final_result[seq_id]={}
        #     final_result[seq_id]['interaction']=ave_activities

        #     ave_bbox_actor=torch.stack(actors_result_seq_collection[seq_id]) # -1, N
        #     ave_bbox_actor=torch.mean(ave_bbox_actor,dim=0) # N
        #     # ave_bbox_actor=torch.argmax(ave_bbox_actor,dim=1) # bbox_num
        #     ave_bbox_action_t=torch.zeros_like(ave_bbox_actor)
        #     # ave_bbox_action_t[torch.topk(ave_bbox_actor,2)[1]]=1
        #     ave_bbox_action_t.scatter_(0,torch.topk(ave_bbox_actor,2,dim=0)[1],1)
        #     final_result[seq_id]['bbox_num']=ave_bbox_actor.shape[0]
        #     ave_bbox_action_t=ave_bbox_action_t.tolist()
        #     # if len(ave_bbox_actor)<cfg.num_boxes:
        #     #     ave_bbox_actor.append(0)
        #     final_result[seq_id]['actions']=ave_bbox_action_t
        # saveDict(final_result, join(cfg.result_path,'result_epoch%d.json'%epoch))
        torch.save(final_result,join(cfg.result_path,'result_epoch%d.pt'%epoch))
    else:
        final_result=torch.load(cfg.save_result_path)
    activities_acc=evaluateDistant(cfg,final_result,fast=False) # Instead, its MHIA
    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss':-1,
        'activities_acc': activities_acc*100,
    }
    return test_info

def bce(source,target,positive_weight=1,attentions=1):
    return (-1*attentions*(target*torch.log(source+1e-6)*positive_weight+(1-target)*torch.log(1-source+1e-6))).mean()

# from reference.SwiftDPP.display.visdomEnhance import Visdom_E

def train_distant_stage2(data_loader, model, device, optimizer, epoch, cfg,scaler,vis=None):
    ablation=cfg.ablation
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    actors_meter=AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    # action_weight=[0.1,1.50362131, 2.89916957, 2.6973297 , 3.95882893, 1.        ,6.09746434]
    # activity_weight=[1.50362131, 2.89916957, 2.6973297 , 3.95882893, 1.        ,6.09746434]
    action_weight=[0.1]+[10]*6
    activity_weight=[1,]*6
    actor_weight=[0.1,1]
    isfirst=True
    nodes_emb_loss_f=nn.TripletMarginLoss(margin=1.0)
    for batch_data in data_loader:
        model.train()
        optimizer.zero_grad()
        # model.apply(set_bn_eval)

        # prepare batch data
        batch_data = [b.to(device=device) for b in batch_data]
        batch_size = batch_data[0].shape[0]*3
        num_frames = batch_data[0].shape[1]//3

        activities_in = batch_data[3].reshape((batch_size, num_frames))
        actions_in = batch_data[2].reshape(
            (batch_size, num_frames, cfg.num_boxes)).reshape(-1)
        actors_in = batch_data[7].reshape(batch_size, num_frames, cfg.num_boxes,cfg.num_boxes)

        # forward
        # if cfg.training_stage==1:
        #     actions_scores, activities_scores = model(
        #         (batch_data[0], batch_data[1], batch_data[4]))
        # else:
        with autocast():
            actions_scores,activities_scores, actor_scores,nodes_emb = model(
                    (batch_data[0], batch_data[1], batch_data[4], batch_data[5], actors_in), ablation)
            # if ablation:
            #     actions_scores,activities_scores, actor_scores,nodes_emb = model(
            #         (batch_data[0], batch_data[1], batch_data[4], batch_data[5], actors_in), ablation)
            # else:
            #     actions_scores,activities_scores, actor_scores,nodes_emb = model(
            #         (batch_data[0], batch_data[1], batch_data[4], batch_data[5]))
            
            # actions_scores=actions_in
            
            nodes_emb=nodes_emb.reshape((batch_size//3,3)+nodes_emb.shape[1:]) # b, 3, -1 : origin, positive, negative

            bboxes_num = batch_data[4].reshape(batch_size, num_frames)[:,0]
            # attentions=batch_data[7].reshape(batch_size,num_frames) # B, T
            # attentions_ave=torch.mean(attentions,dim=1,keepdim=True)
            # attentions_boxes = attentions.unsqueeze(2).repeat((1, 1, cfg.num_boxes)).reshape(-1,1)
            # attentions_noframes=attentions_ave.repeat((1,cfg.num_boxes)).reshape(-1)

            # Here, we use all actors include the blank actors in measurement in training.
            # But, in testing, we do not include blank actors in measurement.
            # actions_in_nopad = []
            # for b in range(batch_size):
            #     N = bboxes_num[b]
            #     actions_in_nopad.append(actions_in[b][:N])
            # actions_in = torch.cat(actions_in_nopad, dim=0).reshape(-1, )  # ALL_N,

            # activities_in = activities_in[:].reshape(batch_size, )

            # Predict actions

            # actors_in=(actors_in>0).float() # B*T*N,
            actions_loss = (F.cross_entropy(actions_scores.reshape(-1,cfg.num_actions),
                                            actions_in, weight=torch.Tensor(action_weight).to(device),reduction='none')
                            ).mean()
            # actors_loss=F.binary_cross_entropy(actor_scores.reshape(-1),actors_in)
            actors_loss=bce(actor_scores.reshape(-1),actors_in.reshape(-1),positive_weight=10)
            # actors_labels=torch.argmax(actor_scores,dim=3)
            actors_labels=torch.sum(actor_scores,dim=3) # B,T,N
            # Below is the old solution, not good.
            # actors_labels/=(torch.max(actors_labels,dim=2,keepdim=True)[0]+1e-12) # B,T,N
            # actors_labels[actors_labels>0.5]=1
            # actors_labels[actors_labels<1]=0
            actors_labels_top2=torch.topk(actors_labels,k=2,dim=-1)[1]
            actors_labels=torch.zeros_like(actors_labels).scatter_(dim=-1,index=actors_labels_top2,src=torch.ones_like(actors_labels))
            # actors_gt_labels=torch.argmax(actors_in,dim=3).reshape(-1)
            actors_gt_labels=actors_in.sum(dim=3).reshape(-1)
            # actors_labels=torch.zeros_like(actor_scores.detach())
            # actors_labels[torch.topk(actor_scores,2,dim=1)[1]]=1
            # actors_labels.scatter_(1,torch.topk(actor_scores,2,dim=1)[1],1) # B,N
            actions_labels = torch.argmax(actions_scores.reshape(-1,cfg.num_actions), dim=1)  # B(BT),N
            # actions_labels=actions_in
            actors_correct = torch.sum(
                torch.eq(actors_labels.reshape(-1).int(), actors_gt_labels.int()).float())
            actions_correct = torch.sum(
                torch.eq(actions_labels.reshape(-1).int(),actions_in.int()).float())
            
            # Predict activities
            nodes_emb_loss=nodes_emb_loss_f(nodes_emb[:,0,...],nodes_emb[:,1,...],nodes_emb[:,2,...])

            # if isfirst:
            #     vis.tsneE(nodes_emb.reshape(-1 ,cfg.num_features_gcn).detach().cpu(),activities_in.detach().cpu().squeeze(-1),'tsne')
            #     isfirst=False
            activities_scores=activities_scores.reshape(-1,cfg.num_activities)
            activities_in=activities_in.flatten()
            # print(activities_scores.shape)
            # print(activities_in.shape)
            activities_loss = (F.cross_entropy(activities_scores, activities_in,reduction='none',
                                            weight=torch.Tensor(activity_weight).to(device))).mean()
            activities_labels = torch.argmax(activities_scores, dim=1)  # B,
            activities_correct = torch.sum(
                torch.eq(activities_labels.int(), activities_in.int()).float())

            # Get accuracy
            actors_accuracy = actors_correct.item() / actors_gt_labels.shape[0] # Here, we include blank actors, but not include in testing!
            actions_accuracy=actions_correct.item()/ actions_in.shape[0]
            activities_accuracy = activities_correct.item() / \
                activities_in.shape[0]

            actors_meter.update(actors_accuracy,actors_gt_labels.shape[0])
            actions_meter.update(actions_accuracy, actions_in.shape[0])
            activities_meter.update(activities_accuracy,
                                    activities_in.shape[0])

            # Total loss
            if torch.isnan(activities_loss) or torch.isnan(actors_loss) or torch.isnan(actions_loss) or torch.isnan(nodes_emb_loss):
                print('debug')
            total_loss = activities_loss+actors_loss+cfg.actions_loss_weight * actions_loss+nodes_emb_loss
            loss_meter.update(total_loss.item(), batch_size)

        # Optim
        scaler.scale(total_loss).backward()
        # total_loss.backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        'actions_acc': actions_meter.avg * 100,
        'actors_acc': actors_meter.avg*100
    }
    return train_info

# @profile
def train_distant_stage1(data_loader, model, device, optimizer, epoch, cfg,scaler):
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    action_weight = torch.Tensor([0.1,10, 10, 10, 10, 10, 10]).cuda()
    actor_weight = [0.1, 10]
    for batch_data in data_loader:
        model.train()
        optimizer.zero_grad()
        # model.apply(set_bn_eval)

        # prepare batch data
        batch_data = [b.to(device=device) for b in batch_data]
        batch_size = batch_data[0].shape[0]
        num_frames = batch_data[0].shape[1]
        bboxes_num = batch_data[4].reshape(batch_size, num_frames)[:,0]
        # gt_atts=batch_data[1]
        actions_in = batch_data[2].reshape(
            (batch_size, num_frames, cfg.num_boxes)).reshape(-1)
        activities_in = batch_data[3].reshape((batch_size, num_frames)).reshape(-1)
        if actions_in.max()==1:
            print('debug')
            exit(0)

        # forward
        # if cfg.training_stage==1:
        with autocast():
            actions_scores, activities_scores = model(
                (batch_data[0], batch_data[1], batch_data[4]))
            # else:
            # atts = model(
            #     (batch_data[0], batch_data[1]))

            # att_loss=F.binary_cross_entropy(atts,gt_atts)
            actions_loss = F.cross_entropy(actions_scores, actions_in, weight=action_weight)

            # Predict activities
            # activities_loss = F.cross_entropy(activities_scores, activities_in)

            # Total loss
            total_loss = actions_loss#*10+activities_loss
            loss_meter.update(total_loss.item(), batch_size)

        # Optim
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # total_loss.backward()
        # optimizer.step()

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': 0,
        'actions_acc': 0,
        'actors_acc': 0
    }
    return train_info