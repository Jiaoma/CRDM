from typing import List,Dict
import os
from os.path import join
import cv2
import PIL
import re
import tqdm
import shutil
from functools import partialmethod, reduce
from os import mkdir, listdir
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from copy import deepcopy
import json
import numpy as np

from collections import OrderedDict

def safeMkdir(path:str):
    if not os.path.exists(path):
        os.mkdir(path)

class VideoReader:
    def __init__(self,video_path='',anno_path='') -> None:
        self.info_dict={'video':{'vid':None,'len':-1},
                        'anno':{'stime':[],'anno':[]}}
        if video_path is not '':
            self.video_path=video_path
            self.read_video(video_path)
        if anno_path is not '':
            self.anno_path=anno_path
            self.read_annotation(anno_path)
        if (anno_path is not '') | (video_path is not ''):
            self.print_info()
    

    def read_video(self,video_path:str) -> None:
        '''
        Give me the path to the video file,
        I open the video file and wait for other command.
        '''
        self.info_dict['video']['vid']=cv2.VideoCapture(video_path)
        # Warning!!! Here is a bug in opencv, the length is not correct, in my case, minus 2 is right.
        self.info_dict['video']['len']=int(self.info_dict['video']['vid'].get(cv2.CAP_PROP_FRAME_COUNT))-2

    def read_annotation(self,anno_path:str) -> None:
        '''
        Give me the path to the annotation file,
        I open the annotation file and wait for other command. 
        '''
        with open(anno_path,'r',encoding='utf-16') as file:
            line=file.readline()
            while line:
                line=file.readline()
                line=line[:-1]
                if line:
                    ids,anno=re.split('\s+',line)[1:-1]
                    ids=ids.split(':')
                    h,m,s,f=[int(i) for i in ids]
                    frame_id=(h*3600+m*60+s)*30+f
                    print(frame_id,anno)
                    self.info_dict['anno']['stime'].append(frame_id)
                    self.info_dict['anno']['anno'].append(anno)

    def print_info(self) -> Dict:
        '''
        Print all the information I get from both the 
        video and annotations. Then I will return a 
        dict contains these information indexed by the 
        item name.
        '''
        print('Annotation examples:')
        if len(self.info_dict['anno']['stime'])>10:
            print(self.info_dict['anno']['anno'][-10:])
        else:
            print(self.info_dict['anno']['anno'])
        print('Video status:')
        print(self.info_dict['video']['vid'])
        print(self.info_dict['video']['len'])

class MultiVideoSolver:
    def __init__(self,video_readers:List[VideoReader]) -> None:
        '''
        Give me multiple video readers, I aggregate them in myself.
        '''
        self.video_readers=video_readers

    def split_all_video_by_annotations(self,split_item:str,anno_item:str,save_dir:str) -> None:
        '''
        Give me the split item and anno item, I will split all the videos
        which have the same length into clips according to the split item, e.g.,
        time (Only support time now). Each clip will save into dependent folder 
        and named by the anno item plus index number with _, xxxx_0_001.jpeg, e.g. 
        '''
        safeMkdir(save_dir)
        get_fid_anno=lambda sid,camera_i:[self.video_readers[camera_i].info_dict['anno'][split_item],self.video_readers[camera_i].info_dict['anno'][anno_item]]
        seq_counter=0 # Used for determine folder name
        oneround_seq_counter=0 # Used for determine anno
        for camera_i,video_reader in enumerate(tqdm.tqdm(self.video_readers)):
            frame_counter=0 # for name
            oneround_seq_counter=0
            seq_num=len(self.video_readers[camera_i].info_dict['anno'][split_item])
            while(True):
                hasFrame,frame=video_reader.info_dict['video']['vid'].read()
                if hasFrame:
                    if frame_counter<self.video_readers[camera_i].info_dict['anno'][split_item][0]:
                        frame_counter+=1
                        continue
                    else:
                        anno=self.video_readers[camera_i].info_dict['anno'][anno_item][oneround_seq_counter]
                        seq_path=join(save_dir,'seq%d'%seq_counter)
                        safeMkdir(seq_path)
                        image_path=join(seq_path,anno+'_'+str(frame_counter)+'.jpg')
                        if not os.path.exists(image_path):
                            cv2.imwrite(image_path,frame)
                        if frame_counter>= \
                            self.video_readers[camera_i].info_dict['anno'][split_item][oneround_seq_counter] \
                            and oneround_seq_counter<(seq_num-1) and frame_counter< \
                            (self.video_readers[camera_i].info_dict['anno'][split_item][oneround_seq_counter+1]-1) \
                            :                                
                            frame_counter+=1
                        else:
                            if frame_counter>= \
                                self.video_readers[camera_i].info_dict['anno'][split_item][oneround_seq_counter] \
                                and oneround_seq_counter==(seq_num-1) and frame_counter<(self.video_readers[camera_i].info_dict['video']['len']-1):
                                # In the last seq, normally add new frames.
                                frame_counter+=1
                            else:
                                # Not in the last seq, but in the last frame of each seq.
                                # Or in the last seq and in the last frame, the oneround_seq_counter
                                # is set to zero in the begining of each camera.
                                frame_counter+=1
                                oneround_seq_counter+=1
                                seq_counter+=1
                else:
                    break

    
    def random_split_train_test_set(self,anno_item:str,splits_source_dir:str,train_vs_test_ratio:float,save_dir:str) -> None:
        '''
        generate a train-test split, and statistic the number of each actions.
        '''
        pass

class SeqsManager:
    def __init__(self, seqs_path:str) -> None:
        self.seqs_path=seqs_path
        self.seqs_name=[]
        self.seqs_num=-1
        self.actions={} # cate:{'seqs':[],'frame_num':0,'seqs_num':0}

    def get_all_seqs(self):
        # Read all seqs
        # Analyze the actions they contains.
        self.seqs_name=os.listdir(self.seqs_path)
        self.seqs_num=len(self.seqs_name)
        for seq in self.seqs_name:
            frames=os.listdir(join(self.seqs_path,seq))
            one_frame=frames[0]
            if one_frame.startswith('wasted'):
                action='wasted'
            else:
                action=one_frame.split('_')[2]
            if action not in self.actions:
                self.actions[action]={'seqs':[],'frame_num':0,'seqs_num':0}
            self.actions[action]['seqs'].append(seq)
            self.actions[action]['frame_num']+=len(frames)
            self.actions[action]['seqs_num']+=1
        ava_seq_num=0
        ava_frame_num=0
        for action in self.actions.keys():
            print('%s:'%action)
            print('Seq num: %d, Frame num: %d'%(self.actions[action]['seqs_num'],self.actions[action]['frame_num']))
            if action!='wasted' and action!='give':
                ava_seq_num+=self.actions[action]['seqs_num']
                ava_frame_num+=self.actions[action]['frame_num']

        print('Ava seq num:%d, frame num:%d'%(ava_seq_num,ava_frame_num))

    def sort_seqs_by_action(self,save_dir:str):
        # For each action, move its seqs into the same folder.
        safeMkdir(save_dir)
        for action in self.actions.keys():
            action_path=join(save_dir,action)
            safeMkdir(action_path)
            for seq in self.actions[action]['seqs']:
                shutil.copytree(join(self.seqs_path,seq),join(action_path,seq))

    def remove_lowest(self,save_path:str,action:str,remove_num:int):
        seqs_path=join(save_path,action)
        seqs_name=os.listdir(seqs_path)
        frame_nums=[len(os.listdir(join(seqs_path,i))) for i in seqs_name]
        keys=sorted(range(len(frame_nums)),key=lambda k:frame_nums[k]) # Small to big
        for key in keys[:remove_num]:
            shutil.rmtree(join(seqs_path,seqs_name[key]))

class FinalSolver:
    def __init__(self,used_actions:list) -> None:
        self.name_to_abc={'psd':'a','gyy':'b','gym':'b','yhm':'c','wym':'d','zjw':'e','wjz':'f','hrz':'g','wy':'h','wff':'i'}
        self.name_to_digit={'psd':'1','gyy':'2','gym':'2','yhm':'3','wym':'4','zjw':'5','wjz':'6','hrz':'7','wy':'8','wff':'9'}
        self.digit_to_abc={'1':'a','2':'b','3':'c','4':'d','5':'e','6':'f','7':'g','8':'h','9':'i'}
        self.abc_to_digit={'a':'1','b':'2','c':'3','d':'4','e':'5','f':'6','g':'7','h':'8','i':'9'}
        self.debug_digit_to_name={'1':'psd','2':'gy','3':'yhm','4':'wym','5':'zjw','6':'wjz','7':'hrz','8':'wy_','9':'wff'}
        self.used_actions=used_actions
        self.action_seqs_dict={}
        self.save_path=''
        for action in used_actions:
            self.action_seqs_dict[action]={'seqs':[],'seqs_num':0,'frame_num':0}

    def get_all_action_seqs(self,save_path:str):
        self.save_path=save_path
        for action in self.used_actions:
            seqs_path=join(save_path,action)
            seqs_name=os.listdir(seqs_path)
            self.action_seqs_dict[action]['seqs']=seqs_name
            self.action_seqs_dict[action]['seqs_num']=len(seqs_name)
            frame_num=reduce(lambda x,y:x+y,[len(os.listdir(join(seqs_path,i))) for i in seqs_name])
            self.action_seqs_dict[action]['frame_num']=frame_num

        ava_seq_num=0
        ava_frame_num=0
        for action in self.action_seqs_dict.keys():
            print('%s:'%action)
            print('Seq num: %d, Frame num: %d'%(self.action_seqs_dict[action]['seqs_num'],self.action_seqs_dict[action]['frame_num']))
            if action!='wasted' and action!='give':
                ava_seq_num+=self.action_seqs_dict[action]['seqs_num']
                ava_frame_num+=self.action_seqs_dict[action]['frame_num']

        print('Ava seq num:%d, frame num:%d'%(ava_seq_num,ava_frame_num))

    def pick_1_in_5(self,new_save_dir:str):
        safeMkdir(new_save_dir)
        self.new_save_dir=new_save_dir
        frame_num=0
        action_frame_num={}
        for action in self.used_actions:
            action_frame_num[action]=0
        for action in self.used_actions:
            action_path=join(self.save_path,action)
            new_action_path=join(new_save_dir,action)
            safeMkdir(new_action_path)
            seqs_path=[i for i in self.action_seqs_dict[action]['seqs']]
            for seq in seqs_path:
                new_seq_path=join(new_action_path,seq)
                seq_path=join(action_path,seq)
                safeMkdir(new_seq_path)
                frames=os.listdir(seq_path)
                if len(frames)%5==1:
                    selected_frames=frames[::5]
                else:
                    selected_frames=frames[::5]+[frames[-1]]
                for frame in selected_frames:
                    if not os.path.exists(join(new_seq_path,frame)):
                        # print('find one')
                        # print(join(new_seq_path,frame))
                        # shutil.copy(join(seq_path,frame),join(new_seq_path,frame))
                        pass
                    # else:
                    #     print('Strange')
                    #     print(join(new_seq_path,frame))
                    if 'wy_wjz_co' in frame:
                        print('Find')
                        print(seq)
                    frame_num+=1
                    action_frame_num[action]+=1
        print('Total frame num%d'%frame_num)
        for action in self.used_actions:
            print('Action %s, frame num%d'%(action,action_frame_num[action]))
    
    def give_priority(self,save_path:str):
        safeMkdir(save_path)
        DIVIDE_NUM=8
        pro_dict={}
        al=len(self.action_seqs_dict.keys())
        l=len(self.action_seqs_dict[self.used_actions[0]]['seqs'])
        for ai,action in enumerate(self.action_seqs_dict.keys()):
            for si,seq in enumerate(self.action_seqs_dict[action]['seqs']):
                pc=si//DIVIDE_NUM
                num_in_pc=ai*(l//DIVIDE_NUM)+si
                prefix=pc*(al*l//DIVIDE_NUM)+num_in_pc
                pro_dict[seq]='%03d'%prefix
        for action in self.action_seqs_dict.keys():
            for seq in self.action_seqs_dict[action]['seqs']:
                seq_path=join(self.new_save_dir,action,seq)
                for img in os.listdir(seq_path):
                    if not os.path.exists(join(save_path,pro_dict[seq]+'_'+seq+'_'+img)):
                        # shutil.copy(join(seq_path,img),join(save_path,pro_dict[seq]+'_'+seq+'_'+img))
                        pass

    def static_all_files(self,path:str):
        # If all files are in one folder, use this.
        action_num={}
        for action in self.used_actions:
            action_num[action]=0
        
        fnames=os.listdir(path)
        for f in fnames:
            action=f.split('_')[4]
            action_num[action]+=1
        print(action_num)
    
    @staticmethod
    def drawBoundingBox(imgPath:str,bboxes:dict,savePath:str):
        img=cv2.imread(imgPath)
        img=cv2.UMat(img).get()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for key in bboxes.keys():
            bbox=bboxes[key]
            bbox = [int(i) for i in bbox]
            text=str(key)
            f=lambda x:(int(x[0]),int(x[1]))
            cv2.rectangle(img,f(bbox[:2]),f(bbox[2:]), (0, 255, 0), 4)
            cv2.putText(img, text, f(bbox[:2]), font, 2, (0, 0, 255), 3)
        cv2.imwrite(savePath, img)
    
    @staticmethod
    def saveDict(src: dict, save_path: str):
        # Notice that, here the dict is not OrderDict
        with open(save_path, 'w') as f:
            json.dump(src, f)

    @staticmethod
    def readDict(path: str):
        with open(path, 'r') as f:
            d = json.load(f)
        return d

    @staticmethod
    def interpolateBboxes(latter_b: dict, next_b: dict, rate: float) -> dict:
        keys=list(latter_b.keys()) if len(list(latter_b.keys()))<=len(list(next_b.keys())) else list(next_b.keys()) # choose shorter one
        new_dict={}
        for key in keys:
            la=np.asarray(latter_b[key])
            if key not in next_b.keys():
                # print('debug')
                return None
            ne=np.asarray(next_b[key])
            ret=tuple(((ne-la)*rate+la).tolist())
            new_dict[key]=ret
        return new_dict

    # Read xml in pickout
    @staticmethod
    def readFromXml(xml_file):
        name_to_digit={'psd':'1','gyy':'2','gym':'2','yhm':'3','wym':'4','zjw':'5','wjz':'6','hrz':'7','wy':'8','wff':'9'}
        if type(xml_file) == str:
            xml_file = Path(xml_file)
        elif type(xml_file) == Path:
            pass
        else:
            raise TypeError
        assert xml_file.exists(), '{} does not exist.'.format(xml_file)

        tree = ET.parse(xml_file.open())
        img_name = tree.find("filename").text
        bboxes = {}
        objects = tree.findall("object")
        who=[]
        for object in objects:
            person_label = object.find('name').text.lower().strip()
            who.append(person_label)
            bbox = object.find('rectangle')
            xmin = round(float(bbox.find('xmin').text), 2)
            ymin = round(float(bbox.find('ymin').text), 2)
            xmax = round(float(bbox.find('xmax').text), 2)
            ymax = round(float(bbox.find('ymax').text), 2)
            bboxes[person_label] = (xmin, ymin, xmax, ymax)
        # Bug here, due to the annotators often randomly annotate the persons, these order is not correct.
        # Beside, since after python3.6, the default dict is already similar to OrderDict, the keys keep order!.
        prio_index,seq,who1,who2,interaction,fid=img_name.split('.')[0].split('_')
        if (name_to_digit[who1] not in who) or (name_to_digit[who2] not in who):
            print(name_to_digit[who1],name_to_digit[who2], who)
            if len(who)==2 and '1' in who:
                print(xml_file)
            # print('Oh, shit, they must annotate wrong person. The following bug discover way is broken and may not produce correct information. And' 
            # 'I \'m lazy to fix it.') 
        who=[name_to_digit[who1],name_to_digit[who2]]
        return {'name':img_name[4:],'prio':prio_index,'sid':seq,'who':who,'interaction':interaction,'fid':fid,'bboxes':bboxes}

    def _interpolate_1_to_5(self):
        '''
        Run in generateAnno, but can't run indepently. No return, but save a global dict contains all the interpolated annos into self.
        '''
        '''In this version, the disappeared interactor in some frame will be recorded.'''
        for seq in self.seq_xmls_dict.keys():
            new_dict_list=[]
            l=len(self.seq_xmls_dict[seq])
            s=self.seq_xmls_dict[seq]
            for fid in range(0,l-1):
                new_dict_list.append(s[fid])
                c_fid=int(s[fid]['fid'])
                n_fid=int(s[fid+1]['fid'])
                for n_frame in range(c_fid+1,n_fid):
                    n_bboxes=self.interpolateBboxes(s[fid]['bboxes'],s[fid+1]['bboxes'],(n_frame-c_fid)/(n_fid-c_fid))
                    if n_bboxes is None:
                        for d_who in s[fid]['who']:
                            if self.debug_digit_to_name[d_who] not in s[fid]['name']:
                                print(s[fid]['prio']+s[fid]['name'])
                        for d_who in s[fid+1]['who']:
                            if self.debug_digit_to_name[d_who] not in s[fid+1]['name']:
                                print(s[fid+1]['prio']+s[fid+1]['name'])
                        # print('debug')
                        break
                    who=list(n_bboxes.keys())
                    s_seq,who1,who2,interaction,s_fid=s[fid]['name'].split('.')[0].split('_')
                    n_name=s_seq+'_'+who1+'_'+who2+'_'+interaction+'_'+str(n_frame)+'.xml'
                    n_frame_dict={'name':n_name,'prio':s[fid]['prio'],'sid':s[fid]['sid'],'who':who,'interaction':s[fid]['interaction'],
                    'fid':str(n_frame),'bboxes':n_bboxes}
                    new_dict_list.append(n_frame_dict)
            new_dict_list.append(s[l-1])
            self.seq_xmls_dict[seq]=new_dict_list
            self.seq_xmls_dict[seq].sort(key=lambda x:x['fid'])

    def translate_newformat_to_oldformat(self,new_format:dict,first_fid:int):
        old_format_dict={}
        old_format_dict['bboxes'] = {}
        
        who=[]
        for w in new_format['who']:
            who.append(self.digit_to_abc[w])
        _,who1,who2,_,_=new_format['name'].split('.')[0].split('_')
        who_=[self.name_to_abc[who1],self.name_to_abc[who2]]
        n_who=[]
        for w in who_:
            if w in who:
                n_who.append(w)
        who=n_who

        seq_key=new_format['interaction']+"_"+reduce(lambda x,y:x+y,who_)
        old_format_dict['file_name'] = seq_key+'%03d'%(int(new_format['fid'])-first_fid) + '.json'
        old_format_dict['seq_key'] = seq_key
        old_format_dict['frame_id'] = int(new_format['fid'])-first_fid
        old_format_dict['who'] = who
        for w in who:
            if self.abc_to_digit[w] in new_format['bboxes'].keys():
                old_format_dict['bboxes'][w]=new_format['bboxes'][self.abc_to_digit[w]]
        # Plus, for convinience, add one item old name
        temp=new_format['name'].split('.')[0]+'.json' # convert .xml name to .json
        old_format_dict['old_name']=temp[temp.find('_')+1:]
        return old_format_dict

    def generateAnno(self,save_path:str,origin_manual_anno_path:str,img_root_path:str,save_img_path:str,debug_path:str):
        '''
        A compositional function. First generate all the annotations from the orginal manual annotations.
        Then rename all the person names. Save the new annotations into the format of the previous version.
        The action of each frame will be converted. The problem of interactors disappear will be considered.
        For the file name, it recorded not accurate annotation of the interactors, while in the xml file, the 
        interactor annotation should be accurate.
        '''
        safeMkdir(debug_path)
        safeMkdir(save_path)
        safeMkdir(save_img_path)
        self.seq_xmls_dict={}
        self.old_format_dict={} # key: old_name without postfix, value: each frame anno dict
        for xml in os.listdir(origin_manual_anno_path):
            xml_dict=self.readFromXml(join(origin_manual_anno_path,xml))
            if xml_dict['sid'] not in self.seq_xmls_dict.keys():
                self.seq_xmls_dict[xml_dict['sid']]=[]
            self.seq_xmls_dict[xml_dict['sid']].append(xml_dict)
        
        # Sort frames in each seq
        for seq in self.seq_xmls_dict.keys():
            self.seq_xmls_dict[seq].sort(key=lambda x:x['fid'])

        # Interpolate them
        self._interpolate_1_to_5() # DESCRIPTION: {'seq0':{'name':xxx,'xxx':xxx,},}
        for action in self.used_actions:
            print("=========Action:%s=========="%action)
            for seq in self.action_seqs_dict[action]['seqs']:
                print('=========%s========'%seq)
                seq_path=join(save_path,seq)
                safeMkdir(seq_path)
                safeMkdir(join(save_img_path,seq))
                safeMkdir(join(debug_path,seq))
                # Translate into the old format annotation + rename
                first_fid=int(self.seq_xmls_dict[seq][0]['fid'])
                for s_f_i in range(len(self.seq_xmls_dict[seq])):
                    old_format_dict=self.translate_newformat_to_oldformat(self.seq_xmls_dict[seq][s_f_i],first_fid)
                    old_name=old_format_dict['old_name'].split('.')[0]
                    self.old_format_dict[old_name]=old_format_dict
                    # Save into dir + rename, also rename the image name.
                    if not os.path.exists(join(save_img_path,seq,old_format_dict['file_name'].split('.')[0]+'.jpg')):
                        shutil.copy(join(img_root_path,action,seq,old_name+'.jpg'),join(save_img_path,seq,old_format_dict['file_name'].split('.')[0]+'.jpg'))
                    if not os.path.exists(join(seq_path,old_format_dict['file_name'])):
                        self.saveDict(old_format_dict,join(seq_path,old_format_dict['file_name']))
                    if not os.path.exists(join(debug_path,seq,old_name+'.jpg')):
                        self.drawBoundingBox(join(img_root_path,action,seq,old_name+'.jpg'),old_format_dict['bboxes'],join(debug_path,seq,old_name+'.jpg'))

def bug_fix_v1(frames_path:str,annotations_path:str):
    abc_to_digit={'a':'1','b':'2','c':'3','d':'4','e':'5','f':'6','g':'7','h':'8','i':'9'}
    name_to_digit={'psd':'1','gyy':'2','gym':'2','yhm':'3','wym':'4','zjw':'5','wjz':'6','hrz':'7','wy':'8','wff':'9'}
    name_to_abc={'psd':'a','gyy':'b','gym':'b','yhm':'c','wym':'d','zjw':'e','wjz':'f','hrz':'g','wy':'h','wff':'i'}
    seqs_names=os.listdir(annotations_path)
    for seq in seqs_names:
        anno_seq_path=join(annotations_path,seq)
        frame_seq_path=join(frames_path,seq)
        # Read all anno
        xmls_names=os.listdir(anno_seq_path)
        xmls=[FinalSolver.readDict(join(anno_seq_path,xml)) for xml in xmls_names]
        # Extract old frame inde
        extract_f=lambda x:int(x['old_name'].split('.')[0].split('_')[3])
        xml_old_indexs=[extract_f(x) for x in xmls]
        # Find the minist
        mini=min(xml_old_indexs)
        # Change fid to x-minist
        for i,xml in enumerate(xmls):
            xml['frame_id']=xml_old_indexs[i]-mini
            old_name=xml['old_name']
            who1,who2,action,fid=old_name.split('.')[0].split('_')
            xml['who']=[name_to_abc[who1],name_to_abc[who2]]
            new_bboxes={}
            for x in xml['who']:
                _x=abc_to_digit[x]
                if _x in xml['bboxes'].keys():
                    new_bboxes[x]=xml['bboxes'][_x]
            xml['bboxes']=new_bboxes
            xml['seq_key']=action+'_'+xml['who'][0]+xml['who'][1]
            xml['file_name']=action+'_'+xml['who'][0]+xml['who'][1]+'%03d'%xml['frame_id']+'.json'
            # Change anno name, change image name,save
            os.remove(join(anno_seq_path,xmls_names[i]))
            FinalSolver.saveDict(xml,join(anno_seq_path,xml['file_name']))
            os.rename(join(frame_seq_path,xmls_names[i].split('.')[0]+'.jpg'),join(frame_seq_path,xml['file_name'].split('.')[0]+'.jpg'))

def extract_txt(results_path:str):
    for seq in os.listdir(results_path):
        shutil.move(join(results_path,seq,'results.txt'),join(results_path,seq+'.txt'))
        # shutil.rmtree(join(results_path,seq))

def find_max_num(results_path:str,resize_HW:tuple,ori_sizes:tuple):
    all_seq_id_bbox_dict={}
    seq_max_bbox_num_dict={}
    '''
    Dict{
    key: seq (int)
    Value: 
            List[
                Dict{
                key:pid (int)
                Value: # bbox
                        (ymin,xmin,ymax,xmax)
                }
            ]
    }
    
    Plus:
    Reindex the id given by the MOT method. Since the given ids are not continous in number.
    '''
    msid,mfid=0,0
    seq_names=os.listdir(results_path)
    seq_names.sort(key=lambda x:int(x.split('.')[0][3:]))
    max_max_n=0
    
    for seq in seq_names:
        seq_oid_nid={}
        on_counter=0
        seq_int=int(seq.split('.')[0][3:])
        with open(join(results_path,seq),'r') as f:
            all_seq_id_bbox_dict[seq_int]=[]
            max_n=0
            for line in f:
                result=line.split(',')
                fid=int(result[0])-1 #Since it starts at 1
                if fid>=len(all_seq_id_bbox_dict[seq_int]):
                    all_seq_id_bbox_dict[seq_int].append({})
                pid=int(result[1])
                if pid not in seq_oid_nid.keys():
                    seq_oid_nid[pid]=on_counter
                    on_counter+=1
                pid=seq_oid_nid[pid]
                max_n=max_n if max_n>=pid else pid
                xmin,ymin,w,h=[max(float(i),0) for i in result[2:6]]
                xmax=xmin+w
                ymax=ymin+h
                xmin = round(xmin * resize_HW[1] / ori_sizes[1], 2)
                ymin = round(ymin * resize_HW[0] / ori_sizes[0], 2)
                xmax = round(xmax * resize_HW[1] / ori_sizes[1], 2)
                ymax = round(ymax * resize_HW[0] / ori_sizes[0], 2)
                H, W = resize_HW
                all_seq_id_bbox_dict[seq_int][fid][pid]=(ymin / H, xmin / W, ymax/ H, xmax / W)
            seq_max_bbox_num_dict[seq_int]=max_n
            max_max_n=max(max_max_n,max_n)
            if max_n==max_max_n:
                # For debug
                msid=seq
    max_max_n=30
    return all_seq_id_bbox_dict,seq_max_bbox_num_dict,max_max_n
                

if __name__=='__main__':
    # GoPros=[
    #     VideoReader('GoPro1.mpeg','all.txt'),
    #     VideoReader('GoPro2.mpeg','all.txt'),
    #     VideoReader('GoPro3.mpeg','all.txt'),
    # ]
    # solver=MultiVideoSolver(GoPros)
    # solver.split_all_video_by_annotations('stime','anno','./extracted')
    # sm=SeqsManager('.\\extracted')
    # sm.get_all_seqs()
    # sm.sort_seqs_by_action('sorted')
    # sm.remove_lowest('.\\sorted','co',40)
    # sm.remove_lowest('.\\sorted','ga',40)
    # sm.remove_lowest('.\\sorted','tc',2)
    # sm.remove_lowest('.\\sorted','pp',8)
    fs=FinalSolver(['hl','tc','co','ga','pp','ch'])
    fs.get_all_action_seqs('data/DistantDatasetV2/annotations')
    # # fs.pick_1_in_5('.\\picked')
    # # fs.give_priority('.\\picked_prio')
    # # fs.static_all_files('.\\picked_prio')
    # fs.generateAnno('.\\annotation_v2','.\\manual_origin','.\\sorted','.\\frames_v2','.\\debug_v2')
    # # bug_fix_v1('.\\DistantDatasetV2\\frames','.\\DistantDatasetV2\\annotation')
    # extract_txt('/home/molijuly/ExternalDisk2/Datasets_formated/DistantDatasetV2/results')
    # all_seq_id_bbox_dict,seq_max_bbox_num_dict,max_max_n=find_max_num('/home/molijuly/ExternalDisk2/Datasets_formated/DistantDatasetV2/results',(1520,2704),(1520,2704))
    # print(max_max_n)