import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms
# from memory_profiler import profile

import xml.etree.ElementTree as ET

from collections import OrderedDict

import random
from PIL import Image
import numpy as np

from collections import Counter
import os

from os.path import join
import pathlib

import tqdm
import torch

from distant_utils import readDict, safeMkdir,are_same,generate_actors

from dataset_script import find_max_num

import math

# FRAMES_NUM = {0:246,1:279,2:150,3:310,4:183,5:215,6:270,7:156,8:175,9:186,10:229,11:175,12:147,13:99,14:137,15:89,
#               16:124,17:157,18:108,19:128,20:107,21:116,22:128,23:168,24:115,25:123,26:121,27:102,28:115,29:170,
#               30:152,31:134,32:94,33:181,34:131,35:164,36:123,37:90,38:89,39:89,40:104,41:77,42:140,43:67,44:78,
#               45:90,46:105,47:93,48:328,49:361,50:399,51:379,52:224,53:246,54:277,55:262,56:259,57:222,58:215,59:289,
#               60:76,61:64,62:63,63:80,64:62,65:51,66:52,67:50,68:49,69:101,70:60,71:65,}
# Min frames: 49 Max frames: 399

# ACTIONS = ['NA', 'ch','co','ga','hl','pp','tc']
ACTIVITIES = ['ch','co','ga','hl','pp','tc']

ACTIONS=['NA','ch','co','ga','hl','pp','tc']

ATOMIC_ACTION={
    'ch':['ch','ch'],
    'co':['co','co'],
    'ga':['ga','ga'],
    'hl':['hl','hl'],
    'pp':['pp','pp'],
    'tc':['tc','tc']
}


ACTIONS_ID = {a: i for i, a in enumerate(ACTIONS)}
ACTIVITIES_ID = {a: i for i, a in enumerate(ACTIVITIES)}

def convertDict(source:dict):
    ret=OrderedDict()
    keys=sorted(source.keys(),key=lambda x:x)
    for key in keys:
        ret[key]=source[key]
    return ret

def read_process_dict(dict_path: str, resize_HW, ori_sizes=(1520, 2704)):
    '''
    The real id, bbox and interaction is followed by the MOT results.
    Here, we extract the manual annotation dict. In dataset class, we 
    will convert these dicts into the MOT results, then aggregate them
    into a unified dict list.
    '''
    abc_to_digit={'a':'1','b':'2','c':'3','d':'4','e':'5','f':'6','g':'7','h':'8','i':'9'}
    source = readDict(dict_path)
    # Name parse
    dict_name = dict_path.split('/')[-1].split('.')[0]  # like ch_ab000
    interact = ACTIVITIES.index(dict_name[:2])
    # who2 = [i for i in dict_name[3:5]]
    who2=source['who'] # Even two persons in file name, there still be possible to have only one person.
    who1_file=dict_name[3]
    who2_file=dict_name[4]
    who2_for_index=[who1_file,who2_file]
    
    frame_id = int(dict_name[5:])
    attr = {}
    attr['frame_id'] = frame_id
    attr['actions'] = []
    attr['bboxes'] = []
    attr['person_label'] = []
    attr['seq_key'] = dict_name[:5]
    attr['full_name'] = dict_name
    objects = source['bboxes']  # dict
    objects = convertDict(objects)
    for object in objects.items():
        person_label = ord(object[0])-ord('a') # Has order
        bbox = object[1]
        xmin = round(bbox[0] * resize_HW[1] / ori_sizes[1], 2)
        ymin = round(bbox[1] * resize_HW[0] / ori_sizes[0], 2)
        xmax = round(bbox[2] * resize_HW[1] / ori_sizes[1], 2)
        ymax = round(bbox[3] * resize_HW[0] / ori_sizes[0], 2)
        if object[0] not in who2:
            interact_idx = 0
        else:
            who2_index=who2_for_index.index(object[0])
            interact_idx=ACTIONS.index(ATOMIC_ACTION[ACTIVITIES[interact]][who2_index])
            # interact_idx = interact+1
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin
        H, W = resize_HW  # BUG!!!!
        attr['bboxes'].append((y / H, x / W, (y + h) / H, (x + w) / W))

        attr['person_label'].append(person_label)
        # The interaction id is the action id here.
        attr['actions'].append(interact_idx)
    attr['interaction'] = interact
    return attr

def distant_read_annotations(cfg,path, sid,fid_pid_bboxes_dict,max_n,thred=0.01):
    # Notice that the path is already include '/distantX'.
    '''
    Since only at most 2 persons are labeled in manual dict, it need to generate labels
    for every person in the MOT detection results.
    '''
    abc_to_digit={'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8} # Here, starts with zero, other places starts with 1 because of the manual annotations.
    annotations = {} # {fid: {xxx}}

    path = path + '/seq%d/' % sid

    allFramesNames=[i.split('.')[0] for i in os.listdir(path) if i.endswith('.json')]

    frames_num=len(allFramesNames)

    for i in range(frames_num):
        
        # Read xml file
        annotation_dict=read_process_dict(join(path,allFramesNames[i]+'.json'),cfg.image_size,ori_sizes=cfg.ori_image_size)
        frame_id=annotation_dict['frame_id']
        # Handle annotation_dict
        '''
        only 1 or 2 in annotation_dict
        frame_id: v
        actions: add
        bboxes: add
        person_label: add
        seq_key: v
        full_name: v
        interaction: v
        GOAL -> add more items.
        '''
        who1_file=annotation_dict['full_name'][3]
        who2_file=annotation_dict['full_name'][4]
        who2_for_index=[abc_to_digit[who1_file],abc_to_digit[who2_file]]
        new_dict={'frame_id':frame_id,'seq_key':annotation_dict['seq_key'],
        'full_name':annotation_dict['full_name'],'interaction':annotation_dict['interaction'],'actions':[],'bboxes':[],'person_label':[],'detection_failed_num':0}
        person2_personN_index=[]
        person2_personN_origin_index={}
        for pi,person in enumerate(annotation_dict['person_label']):
            p_ind=are_same(annotation_dict['bboxes'][pi],fid_pid_bboxes_dict[frame_id],thred)
            if p_ind!=-1:
                person2_personN_index.append(p_ind)
                person2_personN_origin_index[p_ind]=person
            else:
                new_dict['detection_failed_num']+=1
        for npid in range(max_n):
            person_index=npid if npid in fid_pid_bboxes_dict[frame_id].keys() else -1
            if person_index==-1:
                new_dict['person_label'].append(npid)
                new_dict['actions'].append(0)
                new_dict['bboxes'].append((0,0,0,0))
            else:
                new_dict['person_label'].append(npid)
                new_dict['bboxes'].append(fid_pid_bboxes_dict[frame_id][npid])
                if npid in person2_personN_index:
                    who2_index=who2_for_index.index(person2_personN_origin_index[npid])
                    new_dict['actions'].append(ACTIONS.index(ATOMIC_ACTION[ACTIVITIES[annotation_dict['interaction']]][who2_index]))
                else:
                    new_dict['actions'].append(0)
                # Here, action label=interaction label+1. Current method is not suitable for the directional interaction label. For example come-here+ and come-here-
        # WARNING! Here if the MOT didn't recognize person at the position where gt has annotated person, the manual annotated bounding box will be ignored and don't affect the training and even the testing!!! (I don't care...) So, the metrics are
        # focusing on the accuracy of the actions of actors.
        # WARNING! In this frame, the interactor id may be different than the previous interactor id since the same interactor may change its id due to the MOT method.
        annotations[frame_id]=new_dict

    return annotations

def distant_read_dataset(path, seqs,cfg,save_path=None,force=False):
    if not force and save_path is not None and os.path.exists(save_path):
        r=torch.load(save_path)
        data=r['data']
        seq_max_bbox_num_dict=r['seq']
        return data, seq_max_bbox_num_dict
    # 1. Read results txt
    all_sid_fid_pid_bbox_dict,seq_max_bbox_num_dict,max_max_n=find_max_num(str(pathlib.Path(path).parent / 'results'),cfg.image_size,cfg.ori_image_size)
    if cfg.num_boxes!=max_max_n:
        print('[Config auto-changed] The global max number of bounding boxes in all data is not %d but %d, and is changed'%(cfg.num_boxes,max_max_n))
        cfg.num_boxes=max_max_n
    data = {}
    for sid in seqs:
        # Version 1: consider different number of people in each seq
        data[sid] = distant_read_annotations(cfg,path, sid,all_sid_fid_pid_bbox_dict[sid],seq_max_bbox_num_dict[sid])
        # Version 2: tolerate different number of people in each seq but use unified max number. Since we have a box_num value, so version 1 is better
        # data[sid] = distant_read_annotations(path, sid,all_sid_fid_pid_bbox_dict[sid],max_max_n)
    if not force and save_path is not None:
        r={'data':data,'seq':seq_max_bbox_num_dict}
        torch.save(r,save_path)
    return data,seq_max_bbox_num_dict

def distant_all_frames(anns):
    # return [(s, f) for s in anns for f in anns[s]]
    res=[]
    # First, find who is the first frame in seqx
    for s in anns:
        first_frame_id=sorted(list(anns[s].keys()))[0]
        # print(anns[s].keys())
        # Then, only pick up frame_id < first+frame_num-10
        for f in anns[s]:
            # print(f)
            # print(first_frame_id)
            # if f<first_frame_id+FRAMES_NUM[s]-10:
            res.append((s,f))
    return res

import pickle
class DistantDataset(data.Dataset):
    """
    1. Divide seq
    2. Give all frame its (dx, dy) estimation
    3. Give sample according to src_fid.
    """

    def __init__(self, anns, frames, images_path, image_size, feature_size, num_boxes=13, num_frames=10,
                 is_training=True, is_stage1=False, is_preprocess=True, K=5):
        self.anns = anns # dict {key: int, value: dict{key: frame_id, value:dict{a lot}} }
        self.seq_ids=list(self.anns.keys())
        self.seq_ids.sort()
        self.frames = frames # list [(seq_id,frame_id), ...]
        self.K=K
        frame_id_min=0
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size

        self.num_boxes = num_boxes
        self.num_frames = num_frames

        self.is_training = is_training
        self.is_stage1 = is_stage1

        self.is_preprocess = is_preprocess
        self.preprocess_path = str(
            pathlib.Path(images_path).parent / 'preprocess')
        self.feature_path=str(pathlib.Path(images_path).parent/'features')
        self.atts_path=str(pathlib.Path(images_path).parent/'atts.pickle')

        with open(self.atts_path,'rb') as f:
            self.atts=pickle.load(f) # dict: sid: dict: fid: att

        # Divide seq, get max_snip_len, {key: snip_id, value:list[frame_id]}
        self.anns_seq_divide_dict=OrderedDict()
        for seq_i in self.anns.keys():
            max_len=math.ceil(len(self.anns[seq_i].keys())/K)
            self.anns_seq_divide_dict[seq_i]={'max_snip_len':max_len,'interaction':self.anns[seq_i][0]['interaction'],'max_att_index':[]}
            self.anns_seq_divide_dict[seq_i]['frame_id']={}
            for k_i in range(K):
                self.anns_seq_divide_dict[seq_i]['frame_id'][k_i]=[]
                for fid in self.anns[seq_i].keys():
                    if fid<max_len*(k_i+1) and fid>=max_len*k_i:
                        self.anns_seq_divide_dict[seq_i]['frame_id'][k_i].append(fid)
                self.anns_seq_divide_dict[seq_i]['frame_id'][k_i].sort()

        for seq_i in self.anns.keys():
            for k_i in range(K):
                att_max=0
                max_in=-1
                for a_i in self.anns_seq_divide_dict[seq_i]['frame_id'][k_i]:
                    if att_max<=self.atts[seq_i][a_i]:
                        max_in=a_i
                        att_max=self.atts[seq_i][a_i]
                self.anns_seq_divide_dict[seq_i]['max_att_index'].append(a_i)

        self.interaction_seqs={}
        for ai in range(len(ACTIVITIES)):
            self.interaction_seqs[ai]=[]
            for si in self.anns_seq_divide_dict.keys():
                if self.anns_seq_divide_dict[si]['interaction']==ai:
                    self.interaction_seqs[ai].append(si)



    def __len__(self):
        """
        Return the total number of samples
        """
        if self.is_training:
            return len(self.frames)
        else:
            return len(self.anns.keys())

    def __getitem__(self, index):
        """
        Generate one sample of the dataset
        """
        if self.is_training:
            select_frames = self.get_frames(self.frames[index])
        else:
            select_frames=self.get_frames(index) # here, index is seq index.
        sample = self.load_samples_sequence(select_frames)

        return sample

    def get_offset(self,sid,src_fid):
        snip_id=src_fid//self.anns_seq_divide_dict[sid]['max_snip_len']
        return src_fid - self.anns_seq_divide_dict[sid]['frame_id'][snip_id][0]

    def get_frames(self, frame):
        if self.is_training:
            sid, src_fid = frame
        else:
            sid=self.seq_ids[frame]
            src_fid=0

        if not self.is_stage1:
            if self.is_training:
                # 1.
                sample_offsets = np.random.choice(self.anns_seq_divide_dict[sid]['max_snip_len'], self.K - 1,
                                                  replace=False).tolist()
                last = len(self.anns_seq_divide_dict[sid]['frame_id'][self.K - 1])
                if last == 0:
                    print('debug')
                last_offset = random.randint(0, len(self.anns_seq_divide_dict[sid]['frame_id'][self.K - 1]) - 1)
                sample_offsets.append(last_offset)
                return [(sid, src_fid, self.anns_seq_divide_dict[sid]['frame_id'][k_i][0] + sample_offsets[k_i])
                        for k_i in range(self.K)]

                # # 3.
                # interaction_src_sid = self.anns_seq_divide_dict[sid]['interaction']
                # peer_seqs = self.interaction_seqs[interaction_src_sid]  # Seqs that share same interaction labels.
                # select_frames=[]
                # for ki in range(self.K):
                #     k_sid=peer_seqs[random.randint(0,len(peer_seqs)-1)]
                #     if ki!=(self.K-1):
                #         sample_offset = random.randint(0,self.anns_seq_divide_dict[k_sid]['max_snip_len'])
                #     else:
                #         sample_offset = random.randint(0, len(self.anns_seq_divide_dict[k_sid]['frame_id'][self.K - 1]) - 1)
                #     select_frames.append((k_sid,self.anns_seq_divide_dict[k_sid]['frame_id'][ki][0]+sample_offset,
                #                           self.anns_seq_divide_dict[k_sid]['frame_id'][ki][0]+sample_offset))
                # # print(select_frames)
                # return select_frames

                # 2.
                # l=len(self.anns[sid].keys())//self.K *self.K
                # samples = np.random.choice(len(self.anns[sid].keys()), self.K,replace=True)
                # return [(sid, samples[k_i], samples[k_i] ) for k_i in range(self.K)]
            else:
                if len(self.anns_seq_divide_dict[sid]['max_att_index'])!=self.K:
                    print(self.anns_seq_divide_dict[sid]['max_att_index'])
                    print(sid)
                return [(sid, self.anns_seq_divide_dict[sid]['max_att_index'][k_i], self.anns_seq_divide_dict[sid]['max_att_index'][k_i])
                        for k_i in range(self.K)]
                # snip_id = src_fid // self.anns_seq_divide_dict[sid]['max_snip_len']
                # frame_offset = src_fid - self.anns_seq_divide_dict[sid]['frame_id'][snip_id][0]
                # sample_offsets = [frame_offset for i in range(self.K - 1)]
                # last_offset = frame_offset if frame_offset < len(self.anns_seq_divide_dict[sid]['frame_id'][self.K - 1]) \
                #     else (len(self.anns_seq_divide_dict[sid]['frame_id'][self.K - 1]) - 1)
                # sample_offsets.append(last_offset)
                # return [(sid, src_fid, self.anns_seq_divide_dict[sid]['frame_id'][k_i][0] + sample_offsets[k_i])
                #         for k_i in range(self.K)]
        else:
            return [(sid,src_fid,src_fid)]

    def load_samples_sequence(self, select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        OH, OW = self.feature_size

        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num = []
        bboxes_dxdy=[]
        bboxes_distance_mat=[]
        sids=[]
        attentions=[]
        sids_rxry={}

        for i, (sid, src_fid, fid) in enumerate(select_frames):
            sids.append(sid)
            if sid not in sids_rxry.keys():
                sids_rxry[sid]=(random.randint(-100,100),random.randint(-100,0))
            attentions.append(self.atts[sid][fid])
            # if self.is_preprocess:
            #     img = np.load(self.preprocess_path + '/seq%d/%s%03d.npz' %
            #                   (sid, self.anns[sid][src_fid]['seq_key'], fid))['arr_0']
            # else:
            #     img = Image.open(
            #         self.images_path + '/seq%d/%s%03d.jpg' % (sid, self.anns[sid][src_fid]['seq_key'], fid))
            #
            #     img = transforms.functional.resize(img, self.image_size)
            #     img = np.array(img)
            #
            #     # H,W,3 -> 3,H,W
            #     img = img.transpose(2, 0, 1)
            # img=np.zeros((3,1,1))
            img=torch.load(join(self.feature_path,'seq%d/%s%03d.pth.tar' %
                               (sid, self.anns[sid][src_fid]['seq_key'], fid)))
            images.append(img)

            temp_boxes = []
            temp_dxdy=[]
            # for i_box in range(len(self.anns[sid][src_fid]['bboxes'])):
            #     box=self.anns[sid][src_fid]['bboxes'][i_box]
            #     y1, x1, y2, x2 = box
            #     w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
            #     cw,ch=(w1+w2)/2,(h1+h2)/2
            #     temp_boxes.append((w1, h1, w2, h2))
            #     if src_fid==(len(self.anns[sid].keys())-1):
            #         ny1, nx1, ny2, nx2 = self.anns[sid][src_fid-1]['bboxes'][i_box]
            #         nw1, nh1, nw2, nh2 = nx1 * OW, ny1 * OH, nx2 * OW, ny2 * OH
            #         ncw, nch = (nw1 + nw2) / 2, (nh1 + nh2) / 2
            #         dh, dw = ch - nch, cw - ncw
            #         temp_dxdy.append((dh, dw))
            #     else:
            #         try:
            #             ny1, nx1, ny2, nx2 = self.anns[sid][src_fid + 1]['bboxes'][i_box]
            #         except:
            #             print(sid,src_fid+1,i_box)
            #             exit(0)
            #         nw1, nh1, nw2, nh2 = nx1 * OW, ny1 * OH, nx2 * OW, ny2 * OH
            #         ncw,nch=(nw1+nw2)/2,(nh1+nh2)/2
            #         dh,dw=nch-ch,ncw-cw
            #         temp_dxdy.append((dh,dw))


            for i_box in range(len(self.anns[sid][src_fid]['bboxes'])):
                seq_len=len(self.anns[sid].keys())
                left_fid=src_fid-4 if (src_fid-4)>=0 else 0
                right_fid=src_fid+4 if (src_fid+4)<seq_len else seq_len-1

                box=self.anns[sid][src_fid]['bboxes'][i_box]
                y1, x1, y2, x2 = box
                w1, h1, w2, h2 = x1 * OW+sids_rxry[sid][0], y1 * OH+sids_rxry[sid][1], x2 * OW+sids_rxry[sid][0], y2 * OH+sids_rxry[sid][1]
                cw,ch=(w1+w2)/2,(h1+h2)/2
                temp_boxes.append((w1, h1, w2, h2))

                ly1, lx1, ly2, lx2 = self.anns[sid][left_fid]['bboxes'][i_box]
                lw1, lh1, lw2, lh2 = lx1 * OW+sids_rxry[sid][0], ly1 * OH+sids_rxry[sid][1], lx2 * OW+sids_rxry[sid][0], ly2 * OH+sids_rxry[sid][1]
                lcw, lch = (lw1 + lw2) / 2, (lh1 + lh2) / 2

                ry1, rx1, ry2, rx2 = self.anns[sid][right_fid]['bboxes'][i_box]
                rw1, rh1, rw2, rh2 = rx1 * OW+sids_rxry[sid][0], ry1 * OH+sids_rxry[sid][1], rx2 * OW+sids_rxry[sid][0], ry2 * OH+sids_rxry[sid][1]
                rcw, rch = (rw1 + rw2) / 2, (rh1 + rh2) / 2

                dh, dw = rch - lch, rcw - lcw
                temp_dxdy.append((dh, dw))

            temp_distance_mat=np.zeros((self.num_boxes,self.num_boxes),dtype=np.float)
            for i_box in range(len(temp_boxes)):
                for j_box in range(len(temp_boxes)):
                    cw,ch=(temp_boxes[i_box][0]+temp_boxes[i_box][2])/2,(temp_boxes[i_box][1]+temp_boxes[i_box][3])/2
                    tcw, tch = (temp_boxes[j_box][0] + temp_boxes[j_box][2]) / 2, (
                                temp_boxes[j_box][1] + temp_boxes[j_box][3]) / 2
                    temp_distance_mat[i_box,j_box]=math.sqrt((cw-tcw)**2+(ch-tch)**2)

            temp_actions = self.anns[sid][src_fid]['actions'][:]
            bboxes_num.append(self.seq_max_bbox_num_dict[sid])

            while len(temp_boxes) != self.num_boxes:
                temp_boxes.append((0, 0, 0, 0))
                temp_actions.append(0)
                temp_dxdy.append((0,0))

            bboxes_dxdy.append(temp_dxdy)
            bboxes.append(temp_boxes)
            bboxes_distance_mat.append(temp_distance_mat)
            actions.append(temp_actions)

            activities.append(self.anns[sid][src_fid]['interaction'])

        # images = np.stack(images)
        bboxes_distance_mat=np.stack(bboxes_distance_mat) # n, num_boxes,num_boxes
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes, dtype=np.float).reshape(-1, self.num_boxes, 4)
        try:
            bboxes_dxdy = np.array(bboxes_dxdy, dtype=np.float).reshape(-1, self.num_boxes, 2)
        except:
            print('debug')
            exit(0)
        actions = np.array(actions, dtype=np.int32).reshape(-1, self.num_boxes)

        # convert to pytorch tensor
        # images = torch.from_numpy(images).float()
        images=torch.stack(images)
        bboxes_distance_mat=torch.from_numpy(bboxes_distance_mat).float()
        bboxes = torch.from_numpy(bboxes).float()
        bboxes_dxdy=torch.from_numpy(bboxes_dxdy).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        sids=torch.Tensor(sids)
        att = torch.Tensor(attentions) # T value
        if self.is_stage1:
            return images, att
        else:
            return images, bboxes, actions, activities, bboxes_num, bboxes_distance_mat, bboxes_dxdy, sids, att
class DistantTripletDataset(data.Dataset):
    """
    1. Divide seq
    2. Give all frame its (dx, dy) estimation
    3. Give sample according to src_fid.
    """

    def __init__(self,cfg, anns, frames,seq_max_bbox_num_dict, images_path, image_size, feature_size, num_boxes=13, num_frames=10,
                 is_training=True, is_stage1=False, is_preprocess=True, K=5,num_interactions=6,continuous_frames=4,interval=3):
        self.cfg=cfg
        self.anns = anns # dict {key: int, value: dict{key: frame_id, value:dict{a lot}} }
        self.seq_ids=list(self.anns.keys())
        self.seq_ids.sort()
        self.frames = frames # list [(seq_id,frame_id), ...]
        self.K=K
        self.seq_max_bbox_num_dict=seq_max_bbox_num_dict
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size

        self.num_boxes = num_boxes
        self.num_frames = num_frames
        self.num_interactions=num_interactions
        self.continuous_frames=continuous_frames
        self.interval=interval

        self.is_training = is_training
        self.is_stage1 = is_stage1

        self.is_preprocess = is_preprocess
        self.preprocess_path = str(
            pathlib.Path(images_path).parent / 'preprocess')
        self.feature_path=str(pathlib.Path(images_path).parent/'features')
        # self.atts_path=str(pathlib.Path(images_path).parent/'atts.pickle')

        # with open(self.atts_path,'rb') as f:
            # self.atts=pickle.load(f) # dict: sid: dict: fid: att

        # Divide seq, get max_snip_len, {key: snip_id, value:list[frame_id]}
        self.anns_seq_divide_dict=OrderedDict()
        self.first_frames=[]
        for seq_i in self.anns.keys():
            # if seq_i==52:
            #     print(K)
            #     print(len(self.anns[seq_i].keys()))
            #     print(math.ceil(len(self.anns[seq_i].keys())/K))
            #     exit(0)
            max_len=math.ceil(len(self.anns[seq_i].keys())/K)
            self.anns_seq_divide_dict[seq_i]={'max_snip_len':max_len,'interaction':self.anns[seq_i][0]['interaction'],'max_att_index':[]}
            self.anns_seq_divide_dict[seq_i]['frame_id']={}
            for k_i in range(K):
                self.anns_seq_divide_dict[seq_i]['frame_id'][k_i]=[]
                for fid in self.anns[seq_i].keys():
                    if fid<max_len*(k_i+1) and fid>=max_len*k_i:
                        self.anns_seq_divide_dict[seq_i]['frame_id'][k_i].append(fid)
                        self.anns_seq_divide_dict[seq_i]['frame_id'][k_i].sort()
                        if k_i == 0:
                            self.first_frames.append(
                                (seq_i, fid))

                self.anns_seq_divide_dict[seq_i]['frame_id'][k_i].sort()


        # for seq_i in self.anns.keys():
        #     for k_i in range(K):
        #         att_max=0
        #         max_in=-1
        #         for a_i in self.anns_seq_divide_dict[seq_i]['frame_id'][k_i]:
        #             if att_max<=self.atts[seq_i][a_i]:
        #                 max_in=a_i
        #                 att_max=self.atts[seq_i][a_i]
        #         self.anns_seq_divide_dict[seq_i]['max_att_index'].append(a_i)

        self.interaction_seqs={}
        for ai in range(len(ACTIVITIES)):
            self.interaction_seqs[ai]=[]
            for si in self.anns_seq_divide_dict.keys():
                if self.anns_seq_divide_dict[si]['interaction']==ai:
                    self.interaction_seqs[ai].append(si)


    def __len__(self):
        """
        Return the total number of samples
        """
        # if self.is_training:
        #     return len(self.frames)
        # else:
        #     return len(self.anns.keys())
        return len(self.first_frames)

    def __getitem__(self, index):
        select_frames = self.get_frames(self.first_frames[index])
        # if self.is_training:
        #     select_frames = self.get_frames(self.frames[index])
        #     # K,K,K: origin, positive, negative
        # else:
        #     select_frames=self.get_frames(index) # here, index is seq index.
        ####
        # Here, the selected frames are considered as key frames.
        # Then, to get short-term infor, we add following frames of each key frame.
        ####
        sample = self.load_samples_sequence(select_frames)
        return sample

    def get_offset(self,sid,src_fid):
        snip_id=src_fid//self.anns_seq_divide_dict[sid]['max_snip_len']
        return src_fid - self.anns_seq_divide_dict[sid]['frame_id'][snip_id][0]

    def normal_get_frame(self,sid):
        sample_offsets = np.random.choice(self.anns_seq_divide_dict[sid]['max_snip_len'],
                                          self.K - 1,
                                          replace=True).tolist()
        last = len(self.anns_seq_divide_dict[sid]['frame_id'][self.K - 1])
        if last == 0:
            print(sid)
            print(self.anns_seq_divide_dict[sid])
            print(self.anns_seq_divide_dict[sid]['frame_id'])
            print(self.anns_seq_divide_dict[sid]['frame_id'][self.K - 1])
            print('debug,last=0')
            exit(0)
        last_offset = random.randint(0, len(self.anns_seq_divide_dict[sid]['frame_id'][self.K - 1]) - 1)
        sample_offsets.append(last_offset)
        return [(sid, self.anns_seq_divide_dict[sid]['frame_id'][k_i][0] + sample_offsets[k_i],
                 self.anns_seq_divide_dict[sid]['frame_id'][k_i][0] + sample_offsets[k_i])
                for k_i in range(self.K)]

    def get_frames(self, frame):
        '''
        training: input: sid, src_fid, ouput: ori, pos,neg with cfg.k key frames
        and cfg.k*cfg.cont_num following frames. 3 x k x cont_num (sid, src_fid, fid)
        '''
        sid, src_fid = frame
        # if self.is_training:
        #     sid, src_fid = frame
        # else:
        #     sid=self.seq_ids[frame]
        #     src_fid=0

        if not self.is_stage1:
            if self.is_training:
                interaction=self.anns_seq_divide_dict[sid]['interaction']
                negative_interaction=interaction
                while(negative_interaction==interaction):
                    negative_interaction=random.randint(0,self.num_interactions-1)
                negative_sid=self.interaction_seqs[negative_interaction][
                    random.randint(0,len(self.interaction_seqs[negative_interaction])-1)]
                positive_sid = sid
                while(positive_sid==sid):
                    positive_sid=self.interaction_seqs[interaction][
                    random.randint(0,len(self.interaction_seqs[interaction])-1)]
                # K,K,K
                select_frames=[]
                select_frames+=self.normal_get_frame(sid)
                select_frames+=self.normal_get_frame(positive_sid)
                select_frames+=self.normal_get_frame(negative_sid)
                return select_frames
            else:
                snip_id = src_fid // self.anns_seq_divide_dict[sid]['max_snip_len']
                frame_offset = src_fid - self.anns_seq_divide_dict[sid]['frame_id'][snip_id][0]
                sample_offsets = [frame_offset for i in range(self.K - 1)]
                last_offset = frame_offset if frame_offset < len(self.anns_seq_divide_dict[sid]['frame_id'][self.K - 1]) \
                    else (len(self.anns_seq_divide_dict[sid]['frame_id'][self.K - 1]) - 1)
                sample_offsets.append(last_offset)
                return [(sid, src_fid, self.anns_seq_divide_dict[sid]['frame_id'][k_i][0] + sample_offsets[k_i])
                        for k_i in range(self.K)]
        else:
            return [(sid,src_fid,src_fid)]
    def load_samples_sequence(self,select_frames):
        if self.is_stage1:
            return self.load_samples_sequence_stage1(select_frames)
        else:
            return self.load_samples_sequence_stage2_feature(select_frames)

    def load_samples_sequence_stage1(self,select_frames):
        # Version 2. Give me sid, src_fid,fid. I return you images and boxes, dxdy with itself and its following cfg.continuous_frames.
        OH, OW = self.feature_size

        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num = []
        bboxes_dxdy = []
        sids = []
        detection_failed_num=[]

        for i, (sid, src_fid, fid) in enumerate(select_frames):
            sids.append(sid)
            detection_failed_num.append(self.anns[sid][src_fid]['detection_failed_num'])
            if self.is_preprocess:
                img = np.load(self.preprocess_path + '/seq%d/%s%03d.npz' %
                            (sid, self.anns[sid][src_fid]['seq_key'], fid))['arr_0']
            else:
                img = Image.open(
                    self.images_path + '/seq%d/%s%03d.jpg' % (sid, self.anns[sid][src_fid]['seq_key'], fid))
            
                img = transforms.functional.resize(img, self.image_size)
                img = np.array(img)
            
                # H,W,3 -> 3,H,W
                img = img.transpose(2, 0, 1)
            selected_fid=[src_fid]
            images.append(img)
            temp_boxes = []
            temp_dxdy = []
            temp_actions = self.anns[sid][src_fid]['actions'][:]
            while len(temp_actions) != self.num_boxes:
                temp_actions.append(0)
            for sfid_i, sele_src_fid in enumerate(selected_fid):
                for i_box in range(len(self.anns[sid][sele_src_fid]['bboxes'])):
                    seq_len = len(self.anns[sid].keys())
                    left_fid = sele_src_fid - 4 if (sele_src_fid - 4) >= 0 else 0
                    right_fid = sele_src_fid + 4 if (sele_src_fid + 4) < seq_len else seq_len - 1

                    box = self.anns[sid][sele_src_fid]['bboxes'][i_box]
                    
                    y1, x1, y2, x2 = box
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                    cw, ch = (w1 + w2) / 2, (h1 + h2) / 2
                    temp_boxes.append([w1, h1,w2,h2])

                    ly1, lx1, ly2, lx2 = self.anns[sid][left_fid]['bboxes'][i_box]
                    lw1, lh1, lw2, lh2 = lx1 * OW, ly1 * OH, lx2 * OW, ly2 * OH
                    lcw, lch = (lw1 + lw2) / 2, (lh1 + lh2) / 2

                    ry1, rx1, ry2, rx2 = self.anns[sid][right_fid]['bboxes'][i_box]
                    rw1, rh1, rw2, rh2 = rx1 * OW, ry1 * OH, rx2 * OW, ry2 * OH
                    rcw, rch = (rw1 + rw2) / 2, (rh1 + rh2) / 2

                    dh, dw = rch - lch, rcw - lcw
                    if self.anns[sid][left_fid]['bboxes'][i_box]==(0,0,0,0) or self.anns[sid][right_fid]['bboxes'][i_box]==(0,0,0,0) or box==(0,0,0,0):
                        temp_dxdy.append([0, 0])
                    else:
                        temp_dxdy.append([dh, dw])

                if sfid_i==0:
                    bboxes_num.append(self.seq_max_bbox_num_dict[sid]) # The followers have same bboxes_num as the first one.
                if len(temp_boxes)> self.num_boxes*(1+sfid_i):
                    print('=debug')
                while len(temp_boxes) != self.num_boxes*(1+sfid_i):
                    temp_boxes.append([0, 0,0,0])
                    temp_dxdy.append([0, 0])

            bboxes_dxdy.append(temp_dxdy)
            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            # bboxes_distance_mat.append(temp_distance_mat)


            activities.append(self.anns[sid][src_fid]['interaction'])

        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32) # T
        bboxes = np.array(bboxes, dtype=np.float).reshape(len(select_frames),self.num_boxes,4) # T MAX_N 4
        bboxes_dxdy = np.array(bboxes_dxdy, dtype=np.float).reshape(len(select_frames), self.num_boxes, 2) # T MAX_N 2
        actions = np.array(actions, dtype=np.int32).reshape(len(select_frames), self.num_boxes)# T MAX_N

        # convert to pytorch tensor
        # images = torch.from_numpy(images).float()
        images = np.stack(images) # * T,MAX_N,3,H,W
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float() # *
        bboxes_dxdy = torch.from_numpy(bboxes_dxdy).float() # *
        actions = torch.from_numpy(actions).long()
        actors_ = actions.clamp(0,1)
        actors=generate_actors(actors_) # T,N,N
        activities = torch.from_numpy(activities).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        sids = torch.Tensor(sids)
        detection_failed_num=torch.Tensor(detection_failed_num)
        assert bboxes.min()>=0
        # att = torch.Tensor(attentions)  # T value
        # if self.is_stage1:
        #     return images, att
        # else:
        #     # return images, bboxes, actions, activities, bboxes_num, bboxes_distance_mat, bboxes_dxdy, sids, att
        #     return images, bboxes, actions, activities, bboxes_num, bboxes_dxdy, sids, att
        return images, bboxes, actions, activities, bboxes_num, bboxes_dxdy, sids,actors,detection_failed_num

    def load_samples_sequence_stage2(self,select_frames):
        # Version 2. Give me sid, src_fid,fid. I return you images and boxes, dxdy with itself and its following cfg.continuous_frames.
        OH, OW = self.feature_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        MAX_N = self.cfg.num_boxes
        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num = []
        bboxes_dxdy = []
        # bboxes_distance_mat = []
        sids = []
        # attentions = []
        detection_failed_num=[]

        for i, (sid, src_fid, fid) in enumerate(select_frames):
            sids.append(sid)
            detection_failed_num.append(self.anns[sid][fid]['detection_failed_num'])
            if self.is_preprocess:
                img = np.load(self.preprocess_path + '/seq%d/%s%03d.npz' %
                            (sid, self.anns[sid][fid]['seq_key'], fid))['arr_0']
            else:
                img = Image.open(
                    self.images_path + '/seq%d/%s%03d.jpg' % (sid, self.anns[sid][fid]['seq_key'], fid))
            
                img = transforms.functional.resize(img, self.image_size)
                img = np.array(img)
            
                # H,W,3 -> 3,H,W
                img = img.transpose(2, 0, 1)
            
            # img = torch.load(join(self.feature_path, 'seq%d/%s%03d.pth.tar' %
            #                     (sid, self.anns[sid][src_fid]['seq_key'], fid))).requires_grad_(False)
            # Find followings:
            # Here, if img doesn't have enough followers, we will duplicate the final one including itself to match the number.
            selected_fid=[fid] #BUG!!!
            imgs=[torch.from_numpy(img)]
            last_img_i=0
            # if self.is_training:
            #     follow_fid=sorted([0]+random.sample(range(1,self.interval*(self.continuous_frames-1)+1),self.continuous_frames-1))
            # else:
            #     follow_fid=[self.interval*ci for ci in range(self.continuous_frames)]
            follow_fid = [self.interval * ci for ci in range(self.continuous_frames)]
            for img_i in range(1,self.continuous_frames):
                if os.path.exists(join(self.preprocess_path, 'seq%d/%s%03d.npz' %
                                  (sid, self.anns[sid][fid]['seq_key'], fid+follow_fid[img_i]))):
                    if img_i!=0:
                        imgs.append(
                            torch.from_numpy(
                            np.load(join(self.preprocess_path, 'seq%d/%s%03d.npz' %
                                  (sid, self.anns[sid][fid]['seq_key'], fid+follow_fid[img_i])))['arr_0']
                            ))
                        selected_fid.append(fid+follow_fid[img_i])
                    last_img_i=follow_fid[img_i]
                else:
                    imgs.append(
                        torch.from_numpy(
                        np.load(join(self.preprocess_path, 'seq%d/%s%03d.npz' %
                                                  (sid, self.anns[sid][fid]['seq_key'], fid + last_img_i)))['arr_0'])
                    )
                    selected_fid.append(fid + last_img_i) # cn, maxn, d,k,k
                # if os.path.exists(join(self.feature_path, 'seq%d/%s%03d.pth.tar' %
                #                   (sid, self.anns[sid][src_fid]['seq_key'], fid+follow_fid[img_i]))):
                #     if img_i!=0:
                #         imgs.append(torch.load(join(self.feature_path, 'seq%d/%s%03d.pth.tar' %
                #                   (sid, self.anns[sid][src_fid]['seq_key'], fid+follow_fid[img_i]))))
                #         selected_fid.append(fid+follow_fid[img_i])
                #     last_img_i=follow_fid[img_i]
                # else:
                #     imgs.append(torch.load(join(self.feature_path, 'seq%d/%s%03d.pth.tar' %
                #                                   (sid, self.anns[sid][src_fid]['seq_key'], fid + last_img_i))))
                #     selected_fid.append(fid + last_img_i) # cn, maxn, d,k,k
            # imgs=torch.cat([i_img.reshape(1,MAX_N,D,K,K).permute(1,2,0,3,4) for i_img in imgs],dim=2) # MAX_N,D,continuous_num*1,K,K
            images+=imgs

            temp_boxes = []
            temp_dxdy = []
            temp_actions = self.anns[sid][fid]['actions'][:]
            # if np.asarray(temp_actions).sum()==0:
            #     print('debug')
            while len(temp_actions) != self.num_boxes:
                temp_actions.append(0)
            for sfid_i, sele_src_fid in enumerate(selected_fid):
                for i_box in range(len(self.anns[sid][sele_src_fid]['bboxes'])):
                    seq_len = len(self.anns[sid].keys())
                    left_fid = sele_src_fid - 4 if (sele_src_fid - 4) >= 0 else 0
                    right_fid = sele_src_fid + 4 if (sele_src_fid + 4) < seq_len else seq_len - 1

                    box = self.anns[sid][sele_src_fid]['bboxes'][i_box]
                    
                    y1, x1, y2, x2 = box
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                    cw, ch = (w1 + w2) / 2, (h1 + h2) / 2
                    # j=[bi>=0 for bi in [y1, x1, y2, x2]]
                    # assert not(False in j)
                    temp_boxes.append([w1, h1,w2,h2])

                    ly1, lx1, ly2, lx2 = self.anns[sid][left_fid]['bboxes'][i_box]
                    lw1, lh1, lw2, lh2 = lx1 * OW, ly1 * OH, lx2 * OW, ly2 * OH
                    lcw, lch = (lw1 + lw2) / 2, (lh1 + lh2) / 2

                    ry1, rx1, ry2, rx2 = self.anns[sid][right_fid]['bboxes'][i_box]
                    rw1, rh1, rw2, rh2 = rx1 * OW, ry1 * OH, rx2 * OW, ry2 * OH
                    rcw, rch = (rw1 + rw2) / 2, (rh1 + rh2) / 2

                    dh, dw = rch - lch, rcw - lcw
                    if self.anns[sid][left_fid]['bboxes'][i_box]==(0,0,0,0) or self.anns[sid][right_fid]['bboxes'][i_box]==(0,0,0,0) or box==(0,0,0,0):
                        temp_dxdy.append([0, 0])
                    else:
                        temp_dxdy.append([dh, dw])

                if sfid_i==0:
                    bboxes_num.append(self.seq_max_bbox_num_dict[sid]) # The followers have same bboxes_num as the first one.

                while len(temp_boxes) != self.num_boxes*(1+sfid_i):
                    temp_boxes.append([0, 0,0,0])
                    temp_dxdy.append([0, 0])
            
            bboxes_dxdy.append(temp_dxdy)
            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            # bboxes_distance_mat.append(temp_distance_mat)


            activities.append(self.anns[sid][fid]['interaction'])

        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32) # T
        bboxes = np.array(bboxes, dtype=np.float).reshape(len(select_frames),self.continuous_frames,self.num_boxes,4).transpose(0,2,3,1) # T MAX_N 4 continuous_num
        bboxes_dxdy = np.array(bboxes_dxdy, dtype=np.float).reshape(len(select_frames),self.continuous_frames, self.num_boxes, 2).transpose(0,2,3,1) # T MAX_N 2 continuous_num
        actions = np.array(actions, dtype=np.int32).reshape(len(select_frames), self.num_boxes)# T MAX_N

        # convert to pytorch tensor
        # images = torch.from_numpy(images).float()
        images = torch.stack(images).float().reshape(len(select_frames),self.continuous_frames,3,*self.cfg.image_size) #  T,F,C,H,W
        # assert bboxes.min()>=0
        bboxes = torch.from_numpy(bboxes).float() # *
        # boxes_center = (bboxes[:, :, :2, 0] + bboxes[:, :, 2:, 0]) / 2
        # boxes_exist = boxes_center.sum(dim=2)
        # boxes_exist = boxes_exist > 0
        # mask = boxes_exist.numpy()
        # select = np.nonzero(~mask)
        
        bboxes_dxdy = torch.from_numpy(bboxes_dxdy).float() # *
        actions = torch.from_numpy(actions).long()
        actors_ = actions.clamp(0,1)
        actors=generate_actors(actors_)
        activities = torch.from_numpy(activities).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        sids = torch.Tensor(sids)
        detection_failed_num=torch.Tensor(detection_failed_num)
        # assert bboxes.min()>=0
        # att = torch.Tensor(attentions)  # T value
        # if self.is_stage1:
        #     return images, att
        # else:
        #     # return images, bboxes, actions, activities, bboxes_num, bboxes_distance_mat, bboxes_dxdy, sids, att
        #     return images, bboxes, actions, activities, bboxes_num, bboxes_dxdy, sids, att
        return images, bboxes, actions, activities, bboxes_num, bboxes_dxdy, sids,actors,detection_failed_num
    
    def load_samples_sequence_stage2_feature(self, select_frames):
        # Version 2. Give me sid, src_fid,fid. I return you images and boxes, dxdy with itself and its following cfg.continuous_frames.
        OH, OW = self.feature_size
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        MAX_N = self.cfg.num_boxes
        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num = []
        bboxes_dxdy = []
        # bboxes_distance_mat = []
        sids = []
        # attentions = []
        detection_failed_num = []

        for i, (sid, src_fid, fid) in enumerate(select_frames):
            sids.append(sid)
            detection_failed_num.append(self.anns[sid][fid]['detection_failed_num'])
            # attentions.append(self.atts[sid][fid])
            # img=np.zeros((3,1,1))
            # if self.is_stage1:
            #     if self.is_preprocess:
            #         img = np.load(self.preprocess_path + '/seq%d/%s%03d.npz' %
            #                     (sid, self.anns[sid][src_fid]['seq_key'], fid))['arr_0']
            #     else:
            #         img = Image.open(
            #             self.images_path + '/seq%d/%s%03d.jpg' % (sid, self.anns[sid][src_fid]['seq_key'], fid))

            #         img = transforms.functional.resize(img, self.image_size)
            #         img = np.array(img)

            #         # H,W,3 -> 3,H,W
            #         img = img.transpose(2, 0, 1)
            #     img=torch.Tensor(img).float()
            # else:
            img = torch.load(join(self.feature_path, 'seq%d/%s%03d.pth.tar' %
                                  (sid, self.anns[sid][fid]['seq_key'], fid)))
            # Find followings:
            # Here, if img doesn't have enough followers, we will duplicate the final one including itself to match the number.
            selected_fid = [fid]
            imgs = [img]
            last_img_i = 0
            # if self.is_training:
            #     follow_fid=sorted([0]+random.sample(range(1,self.interval*(self.continuous_frames-1)+1),self.continuous_frames-1))
            # else:
            #     follow_fid=[self.interval*ci for ci in range(self.continuous_frames)]
            follow_fid = [self.interval * ci for ci in range(self.continuous_frames)]
            for img_i in range(1, self.continuous_frames):
                if os.path.exists(join(self.feature_path, 'seq%d/%s%03d.pth.tar' %
                                                          (sid, self.anns[sid][fid]['seq_key'],
                                                           fid + follow_fid[img_i]))):
                    if img_i != 0:
                        imgs.append(torch.load(join(self.feature_path, 'seq%d/%s%03d.pth.tar' %
                                                    (
                                                    sid, self.anns[sid][fid]['seq_key'], fid + follow_fid[img_i]))))
                        selected_fid.append(fid + follow_fid[img_i])
                    last_img_i = follow_fid[img_i]
                else:
                    imgs.append(torch.load(join(self.feature_path, 'seq%d/%s%03d.pth.tar' %
                                                (sid, self.anns[sid][fid]['seq_key'], fid + last_img_i))))
                    selected_fid.append(fid + last_img_i)  # cn, maxn, d,k,k
            imgs = torch.cat([i_img.reshape(1, MAX_N, D, K, K).permute(1, 2, 0, 3, 4) for i_img in imgs],
                             dim=2)  # MAX_N,D,continuous_num*1,K,K
            images.append(imgs)

            temp_boxes = []
            temp_dxdy = []
            temp_actions = self.anns[sid][fid]['actions'][:]
            while len(temp_actions) != self.num_boxes:
                temp_actions.append(0)
            for sfid_i, sele_src_fid in enumerate(selected_fid):
                for i_box in range(len(self.anns[sid][sele_src_fid]['bboxes'])):
                    seq_len = len(self.anns[sid].keys())
                    left_fid = sele_src_fid - 4 if (sele_src_fid - 4) >= 0 else 0
                    right_fid = sele_src_fid + 4 if (sele_src_fid + 4) < seq_len else seq_len - 1

                    box = self.anns[sid][sele_src_fid]['bboxes'][i_box]

                    y1, x1, y2, x2 = box
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                    cw, ch = (w1 + w2) / 2, (h1 + h2) / 2
                    temp_boxes.append([w1, h1, w2, h2])

                    ly1, lx1, ly2, lx2 = self.anns[sid][left_fid]['bboxes'][i_box]
                    lw1, lh1, lw2, lh2 = lx1 * OW, ly1 * OH, lx2 * OW, ly2 * OH
                    lcw, lch = (lw1 + lw2) / 2, (lh1 + lh2) / 2

                    ry1, rx1, ry2, rx2 = self.anns[sid][right_fid]['bboxes'][i_box]
                    rw1, rh1, rw2, rh2 = rx1 * OW, ry1 * OH, rx2 * OW, ry2 * OH
                    rcw, rch = (rw1 + rw2) / 2, (rh1 + rh2) / 2

                    dh, dw = rch - lch, rcw - lcw
                    if self.anns[sid][left_fid]['bboxes'][i_box] == (0, 0, 0, 0) or self.anns[sid][right_fid]['bboxes'][
                        i_box] == (0, 0, 0, 0) or box == (0, 0, 0, 0):
                        temp_dxdy.append([0, 0])
                    else:
                        temp_dxdy.append([dh, dw])

                if sfid_i == 0:
                    bboxes_num.append(
                        self.seq_max_bbox_num_dict[sid])  # The followers have same bboxes_num as the first one.

                while len(temp_boxes) != self.num_boxes * (1 + sfid_i):
                    temp_boxes.append([0, 0, 0, 0])
                    temp_dxdy.append([0, 0])

            bboxes_dxdy.append(temp_dxdy)
            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            # bboxes_distance_mat.append(temp_distance_mat)

            activities.append(self.anns[sid][fid]['interaction'])

        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)  # T
        bboxes = np.array(bboxes, dtype=np.float).reshape(len(select_frames), self.continuous_frames, self.num_boxes,
                                                          4).transpose(0, 2, 3, 1)  # T MAX_N 4 continuous_num
        bboxes_dxdy = np.array(bboxes_dxdy, dtype=np.float).reshape(len(select_frames), self.continuous_frames,
                                                                    self.num_boxes, 2).transpose(0, 2, 3,
                                                                                                 1)  # T MAX_N 2 continuous_num
        actions = np.array(actions, dtype=np.int32).reshape(len(select_frames), self.num_boxes)  # T MAX_N

        # convert to pytorch tensor
        # images = torch.from_numpy(images).float()
        images = torch.stack(images)  # * T,MAX_N,D,continuous_num*1,K,K
        bboxes = torch.from_numpy(bboxes).float()  # *
        bboxes_dxdy = torch.from_numpy(bboxes_dxdy).float()  # *
        actions = torch.from_numpy(actions).long()
        actors_ = actions.clamp_max(1)
        actors = generate_actors(actors_)
        activities = torch.from_numpy(activities).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        sids = torch.Tensor(sids)
        detection_failed_num = torch.Tensor(detection_failed_num)
        # att = torch.Tensor(attentions)  # T value
        # if self.is_stage1:
        #     return images, att
        # else:
        #     # return images, bboxes, actions, activities, bboxes_num, bboxes_distance_mat, bboxes_dxdy, sids, att
        #     return images, bboxes, actions, activities, bboxes_num, bboxes_dxdy, sids, att
        return images, bboxes, actions, activities, bboxes_num, bboxes_dxdy, sids, actors, detection_failed_num

####################################################### Trash code#############################################
class readDistantDataset():
    """
    1. Divide seq
    2. Give all frame its (dx, dy) estimation
    3. Give sample according to src_fid.
    """

    def __init__(self, anns, frames, seq_max_bbox_num_dict,images_path, image_size, feature_size, num_boxes=13, num_frames=10,
                 is_training=True, is_stage1=False, is_preprocess=True, K=5):
        self.anns = anns # dict {key: int, value: dict{key: frame_id, value:dict{a lot}} }
        self.frames = frames # list [(seq_id,frame_id), ...]
        self.K=K
        frame_id_min=0
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size
        self.seq_max_bbox_num_dict=seq_max_bbox_num_dict

        self.num_boxes = num_boxes
        self.num_frames = num_frames

        self.is_training = is_training
        self.is_stage1 = is_stage1

        self.is_preprocess = is_preprocess
        self.preprocess_path = str(
            pathlib.Path(images_path).parent / 'preprocess')

        # Divide seq, get max_snip_len, {key: snip_id, value:list[frame_id]}
        self.anns_seq_divide_dict=OrderedDict()
        for seq_i in self.anns.keys():
            max_len=math.ceil(len(self.anns[seq_i].keys())/K)
            self.anns_seq_divide_dict[seq_i]={'max_snip_len':max_len}
            self.anns_seq_divide_dict[seq_i]['frame_id']={}
            for k_i in range(K):
                self.anns_seq_divide_dict[seq_i]['frame_id'][k_i]=[]
                for fid in self.anns[seq_i].keys():
                    if fid<max_len*(k_i+1) and fid>=max_len*k_i:
                        self.anns_seq_divide_dict[seq_i]['frame_id'][k_i].append(fid)
                self.anns_seq_divide_dict[seq_i]['frame_id'][k_i].sort()


    def load_samples_sequence(self, select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        OH, OW = self.feature_size

        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num = []
        bboxes_dxdy=[]
        bboxes_distance_mat=[]
        sids=[]

        for i, (sid, src_fid, fid) in enumerate(select_frames):
            sids.append(sid)
            # if self.is_preprocess:
            #     img = np.load(self.preprocess_path + '/seq%d/%s%03d.npz' %
            #                   (sid, self.anns[sid][src_fid]['seq_key'], fid))['arr_0']
            # else:
            #     img = Image.open(
            #         self.images_path + '/seq%d/%s%03d.jpg' % (sid, self.anns[sid][src_fid]['seq_key'], fid))
            #
            #     img = transforms.functional.resize(img, self.image_size)
            #     img = np.array(img)
            #
            #     # H,W,3 -> 3,H,W
            #     img = img.transpose(2, 0, 1)
            img=np.zeros((3,1,1))
            images.append(img)

            temp_boxes = []
            temp_dxdy=[]
            # for i_box in range(len(self.anns[sid][src_fid]['bboxes'])):
            #     box=self.anns[sid][src_fid]['bboxes'][i_box]
            #     y1, x1, y2, x2 = box
            #     w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
            #     cw,ch=(w1+w2)/2,(h1+h2)/2
            #     temp_boxes.append((w1, h1, w2, h2))
            #     if src_fid==(len(self.anns[sid].keys())-1):
            #         ny1, nx1, ny2, nx2 = self.anns[sid][src_fid-1]['bboxes'][i_box]
            #         nw1, nh1, nw2, nh2 = nx1 * OW, ny1 * OH, nx2 * OW, ny2 * OH
            #         ncw, nch = (nw1 + nw2) / 2, (nh1 + nh2) / 2
            #         dh, dw = ch - nch, cw - ncw
            #         temp_dxdy.append((dh, dw))
            #     else:
            #         try:
            #             ny1, nx1, ny2, nx2 = self.anns[sid][src_fid + 1]['bboxes'][i_box]
            #         except:
            #             print(sid,src_fid+1,i_box)
            #             exit(0)
            #         nw1, nh1, nw2, nh2 = nx1 * OW, ny1 * OH, nx2 * OW, ny2 * OH
            #         ncw,nch=(nw1+nw2)/2,(nh1+nh2)/2
            #         dh,dw=nch-ch,ncw-cw
            #         temp_dxdy.append((dh,dw))


            for i_box in range(len(self.anns[sid][src_fid]['bboxes'])):
                seq_len=len(self.anns[sid].keys())
                left_fid=src_fid-4 if (src_fid-4)>=0 else 0
                right_fid=src_fid+4 if (src_fid+4)<seq_len else seq_len-1

                box=self.anns[sid][src_fid]['bboxes'][i_box]
                y1, x1, y2, x2 = box
                w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                cw,ch=(w1+w2)/2,(h1+h2)/2
                temp_boxes.append((w1, h1, w2, h2))

                ly1, lx1, ly2, lx2 = self.anns[sid][left_fid]['bboxes'][i_box]
                lw1, lh1, lw2, lh2 = lx1 * OW, ly1 * OH, lx2 * OW, ly2 * OH
                lcw, lch = (lw1 + lw2) / 2, (lh1 + lh2) / 2

                ry1, rx1, ry2, rx2 = self.anns[sid][right_fid]['bboxes'][i_box]
                rw1, rh1, rw2, rh2 = rx1 * OW, ry1 * OH, rx2 * OW, ry2 * OH
                rcw, rch = (rw1 + rw2) / 2, (rh1 + rh2) / 2

                dh, dw = rch - lch, rcw - lcw
                temp_dxdy.append((dh, dw))

            temp_distance_mat=np.zeros((self.num_boxes,self.num_boxes),dtype=np.float)
            for i_box in range(len(temp_boxes)):
                for j_box in range(len(temp_boxes)):
                    cw,ch=(temp_boxes[i_box][0]+temp_boxes[i_box][2])/2,(temp_boxes[i_box][1]+temp_boxes[i_box][3])/2
                    tcw, tch = (temp_boxes[j_box][0] + temp_boxes[j_box][2]) / 2, (
                                temp_boxes[j_box][1] + temp_boxes[j_box][3]) / 2
                    temp_distance_mat[i_box,j_box]=math.sqrt((cw-tcw)**2+(ch-tch)**2)

            temp_actions = self.anns[sid][src_fid]['actions'][:]
            bboxes_num.append(self.seq_max_bbox_num_dict[sid])

            while len(temp_boxes) != self.num_boxes:
                temp_boxes.append((0, 0, 0, 0))
                temp_actions.append(0)
                temp_dxdy.append((0,0))

            bboxes_dxdy.append(temp_dxdy)
            bboxes.append(temp_boxes)
            bboxes_distance_mat.append(temp_distance_mat)
            actions.append(temp_actions)

            activities.append(self.anns[sid][src_fid]['interaction'])

        images = np.stack(images)
        bboxes_distance_mat=np.stack(bboxes_distance_mat) # n, num_boxes,num_boxes
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes, dtype=np.float).reshape(-1, self.num_boxes, 4)
        try:
            bboxes_dxdy = np.array(bboxes_dxdy, dtype=np.float).reshape(-1, self.num_boxes, 2)
        except:
            print('debug')
            exit(0)
        actions = np.array(actions, dtype=np.int32).reshape(-1, self.num_boxes)

        # convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes_distance_mat=torch.from_numpy(bboxes_distance_mat).float()
        bboxes = torch.from_numpy(bboxes).float()
        bboxes_dxdy=torch.from_numpy(bboxes_dxdy).float()
        actions = torch.from_numpy(actions).long() #T,N
        actors_ = actions.clamp_max(1)
        actors_ = generate_actors(actors_)
        actors=torch.diag_embed(actors_,dim1=-2,dim2=-1) # T,N,N
        activities = torch.from_numpy(activities).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        sids=torch.Tensor(sids)

        return images, bboxes, actions, activities, bboxes_num,bboxes_distance_mat,bboxes_dxdy,sids,actors

def preprocess(images_path, image_size): 
    new_path = pathlib.Path(images_path).parent/'preprocess'
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    seqs = os.listdir(images_path)
    for seq in tqdm.tqdm(seqs):
        safeMkdir(join(new_path, seq))
        img_names = [_i for _i in os.listdir(
            join(images_path, seq)) if _i.endswith('.jpg')]
        for img_name in img_names:
            img = Image.open(join(images_path, seq, img_name))
            img = transforms.functional.resize(img, image_size)
            img = np.array(img)
            # H,W,3 -> 3,H,W
            img = img.transpose(2, 0, 1)
            img = img.astype(np.float16)
            np.savez(join(new_path, seq, img_name.split('.')[0]+'.npz'), img)


def read_seq_fids(path, sid):
    abc_to_digit={'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8} # Here, starts with zero, other places starts with 1 because of the manual annotations.
    annotations = {} # {fid: {xxx}}

    path = path + '/seq%d/' % sid

    names=os.listdir(path)
    names.sort()
    allFramesNames=[i.split('.')[0] for i in names if i.endswith('.json')]
    # TODO: Error

    frames_num=len(allFramesNames)

    for i in range(frames_num):
        # Read xml file
        annotation_dict=read_process_dict(join(path,allFramesNames[i]+'.json'),(480,720),ori_sizes=(1088,1920))
        annotations[i]=annotation_dict
    return annotations

if __name__=='__main__':
    # read seqs, know which of them are train/test co/ga
    train_seqs=[4, 15, 16, 17, 21, 25, 41, 42, 43, 45, 46, 48, 52, 58, 60, 63, 64, 66, 67, 69, 71, 73, 75, 310, 312, 317, 318, 321, 322, 323, 328, 330, 331, 333, 340, 345, 347, 349, 358, 360, 363, 367, 371, 373, 375, 377, 380, 385, 386, 616, 621, 625, 626, 627, 629, 637, 643, 651, 654, 656, 657, 658, 667, 668, 669, 670, 672, 673, 677, 678, 679, 681, 682, 685, 686, 687, 689, 691, 692, 693]
    test_seqs=[6, 9, 10, 11, 13, 14, 19, 20, 22, 23, 24, 27, 31, 34, 38, 39, 44, 53, 54, 55, 56, 57, 61, 65, 70, 74, 77, 79, 80, 81, 315, 316, 319, 320, 325, 326, 327, 329, 337, 344, 348, 350, 351, 354, 359, 361, 362, 364, 366, 369, 370, 372, 376, 379, 381, 383, 387, 618, 622, 623, 624, 628, 631, 632, 633, 634, 635, 636, 639, 646, 650, 653, 655, 660, 664, 665, 666, 675, 676, 683]
    # read their boxes, regardless of the one lack one person
    # Luckily, all of them are two-persons seqs.
    # only need box and interaction-1.
    train_dict={}
    test_dict={}
    # seqid: {interaction:int, fid:{fid:[(bboxes),(),...]}}
    center=lambda y:[((x[0]+x[2])/2,(x[1]+x[3])/2) for x in y]
    for seq in train_seqs:
        a=read_seq_fids('/home/lijiacheng/code/HIT/data/DistantDatasetV2/annotations',seq)
        newdict={'label':a[0]['interaction']-1,'fid':{}}
        for key in a.keys():
            newdict['fid'][key]= center(a[key]['bboxes'])
        train_dict[seq]=newdict
    for seq in test_seqs:
        a=read_seq_fids('/home/lijiacheng/code/HIT/data/DistantDatasetV2/annotations',seq)
        newdict={'label':a[0]['interaction']-1,'fid':{}}
        for key in a.keys():
            newdict['fid'][key]=center(a[key]['bboxes'])
        test_dict[seq]=newdict
    # save to pth
    torch.save(train_dict,'train.pth')
    torch.save(test_dict,'test.pth')