import time
import os


class Config(object):
    """
    class to save config parameter
    """

    def __init__(self, dataset_name):
        # Global
        self.ori_image_size=1088,1920
        self.image_size = 720, 1280  #input image size
        self.batch_size =  32  #train batch size 
        self.test_batch_size = 8  #test batch size
        self.num_boxes = 12  #max number of bounding boxes in each frame
        self.ablation=False
        self.num_workers=12
        
        # Gpu
        self.use_gpu=True
        self.use_multi_gpu=False
        self.device_list="1,"  #id list of gpus used for training
        
        # Dataset
        assert(dataset_name in ['volleyball', 'collective','Distant'])
        self.dataset_name=dataset_name 
        self.force=False
        
        if dataset_name=='volleyball':
            self.data_path='data/volleyball'  #data path for the volleyball dataset
            self.train_seqs = [ 1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54,
                                0,2,8,12,17,19,24,26,27,28,30,33,46,49,51]  #video id list of train set 
            self.test_seqs = [4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47]  #video id list of test set
            
        elif dataset_name=='collective':
            self.data_path='data/collective'  #data path for the collective dataset
            self.test_seqs=[5,6,7,8,9,10,11,15,16,25,28,29]
            self.train_seqs=[s for s in range(1,45) if s not in self.test_seqs]
        else:
            self.data_path='data/DistantDatasetV2'
            self.train_seqs=[9, 10, 11, 14, 19, 21, 24, 27, 38, 41, 42, 44, 48, 54, 55, 56, 57, 60, 63, 65, 70, 71, 73, 74, 79, 85, 86, 91, 92, 93, 98, 99, 101, 106, 107, 108, 112, 113, 114, 115, 116, 117, 123, 130, 131, 135, 138, 141, 142, 146, 150, 152, 153, 154, 155, 157, 160, 161, 167, 171, 179, 180, 181, 182, 184, 187, 188, 191, 193, 197, 201, 202, 205, 209, 212, 216, 220, 224, 228, 229, 231, 234, 240, 246, 247, 250, 254, 263, 265, 268, 270, 273, 277, 279, 280, 282, 284, 285, 289, 290, 298, 300, 301, 303, 304, 315, 316, 321, 322, 326, 327, 329, 331, 333, 347, 348, 349, 350, 354, 359, 360, 363, 373, 376, 377, 379, 381, 383, 385, 387, 392, 394, 402, 404, 405, 407, 410, 413, 415, 419, 421, 423, 426, 429, 431, 432, 433, 434, 436, 439, 441, 443, 444, 446, 449, 453, 457, 458, 460, 462, 466, 467, 470, 471, 475, 478, 480, 483, 485, 486, 488, 490, 492, 496, 503, 507, 509, 515, 518, 522, 523, 525, 526, 530, 534, 538, 546, 549, 553, 558, 563, 566, 568, 569, 571, 573, 574, 576, 578, 582, 585, 586, 590, 591, 596, 600, 607, 608, 609, 611, 616, 625, 627, 629, 631, 632, 636, 637, 646, 650, 653, 656, 657, 660, 665, 667, 669, 670, 672, 673, 676, 677, 679, 681, 682, 683, 686, 689, 692, 693]

            # self.train_seqs=[9, 11, 19, 21, 27, 38, 42, 44, 48, 54, 56, 60, 70, 74, 315, 321, 10, 14, 24, 41, 55, 57, 63, 65, 71, 73, 79, 316, 322, 326, 347, 349, 85, 86, 91, 92, 101, 106, 107, 112, 115, 116, 130, 131, 135, 141, 146, 150, 93, 98, 99, 108, 113, 114, 117, 123, 138, 142, 152, 153, 157, 161, 167, 171, 184, 187, 188, 191, 193, 197, 201, 202, 205, 209, 212, 216, 220, 224, 228, 229, 254, 263, 265, 268, 270, 273, 277, 279, 280, 282, 284, 285, 289, 290, 298, 300] # 0.4 select num: 96

            # self.train_seqs=[9, 11, 19, 21, 27, 38, 42, 44, 48, 54, 56, 60, 70, 74, 315, 321, 327, 329, 331, 333, 348, 350, 354, 360, 10, 14, 24, 41, 55, 57, 63, 65, 71, 73, 79, 316, 322, 326, 347, 349, 359, 363, 373, 377, 379, 381, 383, 385, 85, 86, 91, 92, 101, 106, 107, 112, 115, 116, 130, 131, 135, 141, 146, 150, 154, 155, 160, 179, 180, 392, 402, 407, 93, 98, 99, 108, 113, 114, 117, 123, 138, 142, 152, 153, 157, 161, 167, 171, 181, 182, 394, 404, 405, 410, 415, 419, 184, 187, 188, 191, 193, 197, 201, 202, 205, 209, 212, 216, 220, 224, 228, 229, 231, 234, 240, 246, 247, 250, 490, 492, 254, 263, 265, 268, 270, 273, 277, 279, 280, 282, 284, 285, 289, 290, 298, 300, 301, 303, 304, 558, 563, 566, 568, 569] # 0.6 select num: 144

            # self.train_seqs=[9, 11, 19, 21, 27, 38, 42, 44, 48, 54, 56, 60, 70, 74, 315, 321, 327, 329, 331, 333, 348, 350, 354, 360, 376, 616, 625, 627, 629, 631, 637, 650, 10, 14, 24, 41, 55, 57, 63, 65, 71, 73, 79, 316, 322, 326, 347, 349, 359, 363, 373, 377, 379, 381, 383, 385, 387, 632, 636, 646, 653, 657, 665, 667, 85, 86, 91, 92, 101, 106, 107, 112, 115, 116, 130, 131, 135, 141, 146, 150, 154, 155, 160, 179, 180, 392, 402, 407, 413, 421, 426, 431, 432, 436, 441, 446, 93, 98, 99, 108, 113, 114, 117, 123, 138, 142, 152, 153, 157, 161, 167, 171, 181, 182, 394, 404, 405, 410, 415, 419, 423, 429, 433, 434, 439, 443, 444, 449, 184, 187, 188, 191, 193, 197, 201, 202, 205, 209, 212, 216, 220, 224, 228, 229, 231, 234, 240, 246, 247, 250, 490, 492, 496, 503, 507, 509, 515, 518, 522, 523, 254, 263, 265, 268, 270, 273, 277, 279, 280, 282, 284, 285, 289, 290, 298, 300, 301, 303, 304, 558, 563, 566, 568, 569, 571, 573, 574, 576, 578, 582, 585, 586]
            # # 0.8 select num: 192

            self.test_seqs=[4, 6, 13, 15, 16, 17, 20, 22, 23, 25, 31, 34, 39, 43, 45, 46, 52, 53, 58, 61, 64, 66, 67, 69, 75, 77, 80, 81, 87, 88, 94, 96, 97, 102, 103, 109, 111, 118, 120, 121, 122, 125, 126, 128, 132, 133, 136, 137, 140, 143, 145, 147, 148, 149, 151, 156, 159, 162, 164, 165, 166, 169, 170, 172, 174, 175, 176, 177, 186, 190, 194, 203, 206, 208, 211, 213, 214, 217, 219, 232, 235, 238, 241, 243, 244, 249, 252, 256, 257, 259, 260, 261, 262, 267, 272, 274, 276, 283, 286, 287, 291, 292, 294, 296, 299, 302, 305, 310, 312, 317, 318, 319, 320, 323, 325, 328, 330, 337, 340, 344, 345, 351, 358, 361, 362, 364, 366, 367, 369, 370, 371, 372, 375, 380, 386, 391, 393, 397, 398, 399, 400, 403, 408, 409, 412, 414, 417, 418, 420, 422, 424, 427, 428, 437, 438, 442, 447, 448, 451, 452, 454, 455, 456, 459, 461, 463, 465, 468, 472, 473, 476, 477, 481, 482, 487, 493, 494, 497, 499, 500, 508, 511, 512, 514, 517, 519, 520, 535, 537, 540, 541, 544, 547, 550, 552, 555, 556, 560, 562, 565, 567, 579, 580, 583, 588, 589, 592, 593, 595, 597, 598, 602, 604, 605, 606, 610, 618, 621, 622, 623, 624, 626, 628, 633, 634, 635, 639, 643, 651, 654, 655, 658, 664, 666, 668, 675, 678, 685, 687, 691]
            # self.test_seqs=self.train_seqs
            # self.train_seqs=[0,1,2,3,4,5,12,13,14,15,16,17,24,25,26,27,28,29,36,37,38,39,40,41,48,49,50,51,52,53,60,61,62,63,64,65,]
            # self.test_seqs=[6,7,8,9,10,11,18,19,20,21,22,23,30,31,32,33,34,35,42,43,44,45,46,47,54,55,56,57,58,59,66,67,68,69,70,71,]
            # self.train_seqs=[i*2 for i in range(36)]
            # self.test_seqs=[i*2+1 for i in range(36)]
            # self.test_seqs = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29, 36, 37, 38, 39, 40, 41,
            #                    48, 49, 50, 51, 52, 53, 60, 61, 62, 63, 64, 65, ]
            # self.train_seqs = [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46,
            #                   47, 54, 55, 56, 57, 58, 59, 66, 67, 68, 69, 70, 71, ]
            # self.train_seqs = [0, 1, 2, 3, 4, 5,6, 7, 8, 9, 12, 13, 14, 15, 16, 17,18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,36, 37, 38, 39, 40, 41,
            #                    42, 43, 44, 45,48, 49, 50, 51, 52, 53,54, 55, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,]
            # self.test_seqs = [ 10, 11,  22, 23,
            #                    34, 35,  46, 47,
            #                    58, 59,  70, 71, ]
        # Backbone 
        self.backbone='inv3' 
        self.crop_size = 5, 5  #crop size of roi align
        self.train_backbone = False  #if freeze the feature extraction part of network, True for stage 1, False for stage 2[9, 11, 19, 21, 27, 38, 42, 44, 48, 54, 56, 60, 70, 74, 315, 321, 327, 329, 331, 333, 348, 350, 354, 360, 10, 14, 24, 41, 55, 57, 63, 65, 71, 73, 79, 316, 322, 326, 347, 349, 359, 363, 373, 377, 379, 381, 383, 385, 85, 86, 91, 92, 101, 106, 107, 112, 115, 116, 130, 131, 135, 141, 146, 150, 154, 155, 160, 179, 180, 392, 402, 407, 93, 98, 99, 108, 113, 114, 117, 123, 138, 142, 152, 153, 157, 161, 167, 171, 181, 182, 394, 404, 405, 410, 415, 419, 184, 187, 188, 191, 193, 197, 201, 202, 205, 209, 212, 216, 220, 224, 228, 229, 231, 234, 240, 246, 247, 250, 490, 492, 254, 263, 265, 268, 270, 273, 277, 279, 280, 282, 284, 285, 289, 290, 298, 300, 301, 303, 304, 558, 563, 566, 568, 569]
        self.out_size = 87, 157  #output feature map size of backbone 
        self.emb_features=288 #185   #output feature map channel of backbone
        self.compress_emb_features=8

        
        # Activity Action
        self.num_actions = 7  #number of action categories
        self.num_activities = 6  #number of activity categories
        self.actions_loss_weight = 1  #weight used to balance action loss and activity loss
        self.actions_weights = None

        # Sample
        self.K=5
        self.num_frames = 3 
        self.num_before = 5
        self.num_after = 4
        self.continuous_frames = 5
        self.interval=3

        # GCN
        self.num_features_pose = 6
        self.num_features_trace = 6
        self.num_features_boxes = 32
        self.num_features_relation=256
        self.num_features_relation_app=256
        self.num_features_relation_mt=128
        self.num_graph=16  #number of graphs
        self.num_features_gcn=self.num_features_boxes
        self.gcn_layers=1  #number of GCN layers
        self.tau_sqrt=False
        self.pos_threshold=0.2  #distance mask threshold in position relation

        # Training Parameters
        self.train_random_seed = 0
        self.train_learning_rate = 2e-4  #initial learning rate 
        self.lr_plan = {41:1e-4, 81:5e-5, 121:1e-5}  #change learning rate in these epochs 
        self.train_dropout_prob = 0.8  #dropout probability
        self.weight_decay = 0  #l2 weig[9, 11, 19, 21, 27, 38, 42, 44, 48, 54, 56, 60, 70, 74, 315, 321, 327, 329, 331, 333, 348, 350, 354, 360, 10, 14, 24, 41, 55, 57, 63, 65, 71, 73, 79, 316, 322, 326, 347, 349, 359, 363, 373, 377, 379, 381, 383, 385, 85, 86, 91, 92, 101, 106, 107, 112, 115, 116, 130, 131, 135, 141, 146, 150, 154, 155, 160, 179, 180, 392, 402, 407, 93, 98, 99, 108, 113, 114, 117, 123, 138, 142, 152, 153, 157, 161, 167, 171, 181, 182, 394, 404, 405, 410, 415, 419, 184, 187, 188, 191, 193, 197, 201, 202, 205, 209, 212, 216, 220, 224, 228, 229, 231, 234, 240, 246, 247, 250, 490, 492, 254, 263, 265, 268, 270, 273, 277, 279, 280, 282, 284, 285, 289, 290, 298, 300, 301, 303, 304, 558, 563, 566, 568, 569]ht decay
    
        self.max_epoch=150  #max training epoch
        self.test_interval_epoch=2
        
        # Exp
        self.training_stage=1  #specify stage1 or stage2
        self.stage1_model_path=''  #path of the base model, need to be set in stage2
        self.test_before_train=False
        self.exp_note='Interaction recognition'
        self.exp_name=None
        
        
    def init_config(self, need_new_folder=True):
        if self.exp_name is None:
            time_str=time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name='[%s_stage%d]<%s>'%(self.exp_note,self.training_stage,time_str)
            
        self.result_path='result/%s'%self.exp_name
        self.log_path='result/%s/log.txt'%self.exp_name
            
        if need_new_folder:
            os.mkdir(self.result_path)
            
            
if __name__=='__main__':
    train_seqs=[9, 10, 11, 14, 19, 21, 24, 27, 38, 41, 42, 44, 48, 54, 55, 56, 57, 60, 63, 65, 70, 71, 73, 74, 79, 85, 86, 91, 92, 93, 98, 99, 101, 106, 107, 108, 112, 113, 114, 115, 116, 117, 123, 130, 131, 135, 138, 141, 142, 146, 150, 152, 153, 154, 155, 157, 160, 161, 167, 171, 179, 180, 181, 182, 184, 187, 188, 191, 193, 197, 201, 202, 205, 209, 212, 216, 220, 224, 228, 229, 231, 234, 240, 246, 247, 250, 254, 263, 265, 268, 270, 273, 277, 279, 280, 282, 284, 285, 289, 290, 298, 300, 301, 303, 304, 315, 316, 321, 322, 326, 327, 329, 331, 333, 347, 348, 349, 350, 354, 359, 360, 363, 373, 376, 377, 379, 381, 383, 385, 387, 392, 394, 402, 404, 405, 407, 410, 413, 415, 419, 421, 423, 426, 429, 431, 432, 433, 434, 436, 439, 441, 443, 444, 446, 449, 453, 457, 458, 460, 462, 466, 467, 470, 471, 475, 478, 480, 483, 485, 486, 488, 490, 492, 496, 503, 507, 509, 515, 518, 522, 523, 525, 526, 530, 534, 538, 546, 549, 553, 558, 563, 566, 568, 569, 571, 573, 574, 576, 578, 582, 585, 586, 590, 591, 596, 600, 607, 608, 609, 611, 616, 625, 627, 629, 631, 632, 636, 637, 646, 650, 653, 656, 657, 660, 665, 667, 669, 670, 672, 673, 676, 677, 679, 681, 682, 683, 686, 689, 692, 693]
    test_seqs=[4, 6, 13, 15, 16, 17, 20, 22, 23, 25, 31, 34, 39, 43, 45, 46, 52, 53, 58, 61, 64, 66, 67, 69, 75, 77, 80, 81, 87, 88, 94, 96, 97, 102, 103, 109, 111, 118, 120, 121, 122, 125, 126, 128, 132, 133, 136, 137, 140, 143, 145, 147, 148, 149, 151, 156, 159, 162, 164, 165, 166, 169, 170, 172, 174, 175, 176, 177, 186, 190, 194, 203, 206, 208, 211, 213, 214, 217, 219, 232, 235, 238, 241, 243, 244, 249, 252, 256, 257, 259, 260, 261, 262, 267, 272, 274, 276, 283, 286, 287, 291, 292, 294, 296, 299, 302, 305, 310, 312, 317, 318, 319, 320, 323, 325, 328, 330, 337, 340, 344, 345, 351, 358, 361, 362, 364, 366, 367, 369, 370, 371, 372, 375, 380, 386, 391, 393, 397, 398, 399, 400, 403, 408, 409, 412, 414, 417, 418, 420, 422, 424, 427, 428, 437, 438, 442, 447, 448, 451, 452, 454, 455, 456, 459, 461, 463, 465, 468, 472, 473, 476, 477, 481, 482, 487, 493, 494, 497, 499, 500, 508, 511, 512, 514, 517, 519, 520, 535, 537, 540, 541, 544, 547, 550, 552, 555, 556, 560, 562, 565, 567, 579, 580, 583, 588, 589, 592, 593, 595, 597, 598, 602, 604, 605, 606, 610, 618, 621, 622, 623, 624, 626, 628, 633, 634, 635, 639, 643, 651, 654, 655, 658, 664, 666, 668, 675, 678, 685, 687, 691]
    # from random import shuffle
    # from functools import reduce
    # action_seqs={0:[],1:[],2:[],3:[],4:[],5:[]} 
    # for i in range(240): 
    #     a=i//40 
    #     action_seqs[a].append(train_seqs[i]) 
    #     action_seqs[a].append(test_seqs[i]) 
    
    # action_same_seqs={0:[],1:[],2:[],3:[],4:[],5:[]}
    # for ac in action_seqs.keys():
    #     action_seqs[ac].sort()
    
    # for ac in action_same_seqs.keys():
    #     select=[]
    #     selected=[]
    #     for seq in action_seqs[ac]:
    #         if seq not in selected:
    #             select.append(seq)
    #             selected.append(seq)
    #             if seq+306 in action_seqs[ac] and seq+306 not in selected:
    #                 select.append(seq+306)
    #                 selected.append(seq+306)
    #             if seq+306*2 in action_seqs[ac] and seq+306*2 not in selected:
    #                 select.append(seq+306*2)
    #                 selected.append(seq+306*2)
    #             action_same_seqs[ac].append(select)
    #     shuffle(action_same_seqs[ac])
    #     action_same_seqs[ac]=reduce(lambda x,y:x+y,action_same_seqs[ac])
    # # print(action_same_seqs[5])
    # # print(action_same_seqs[4])
    # import numpy as np 
    # new_train_seqs,new_test_seqs=[],[]
    # # for key in action_same_seqs.keys(): 
    # for key in [0,1,2,3,4,5]: 
    #     actions=action_same_seqs[key] 
    #     indexs=np.random.choice(80,80,replace=False) 
    #     for i in indexs[:30]: 
    #         new_train_seqs.append(actions[i]) 
    #     for i in indexs[40:]: 
    #         new_test_seqs.append(actions[i])
    # new_train_seqs.sort()
    # new_test_seqs.sort()
    # print(new_train_seqs)
    # print(new_test_seqs)

    '''
    Calculate different ratios of training set with balanced class video num.
    '''
    action_seqs={'co':[],'ga':[],'hl':[],'tc':[],'pp':[],'ch':[]} 
    for seq in train_seqs:
        jsonf=os.listdir(os.path.join('/home/lijiacheng/code/HIT/data/DistantDatasetV2/annotations','seq%d/'%seq))
        jsonf1=jsonf[0]
        action_seqs[jsonf1[:2]].append(seq)
    for key in action_seqs.keys():
        print(len(action_seqs[key]))

    ratio=0.1
    selectnum=int(len(train_seqs)*ratio)
    print('select num: %d'%selectnum)
    new_list=[]
    for key in action_seqs.keys():
        for i in range(selectnum//6):
            new_list.append(action_seqs[key][i])
    print(new_list)