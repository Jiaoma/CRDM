'''
Result form:
1. {seq_name: action}
2. {seq_name: this action: who are interacting with each other.} {seq_name: for the given bounding boxes give their
probability of being interactors. Top-2 accuracy.}

seq-action-acc: /frac{num_correct_seq_pred}{num_seq}
seq_actors-acc-top2: /frac{num_top2actors_matched}{num_seq}

file form:
dict{'seq_name': dict{'action':int, 'actors': char(Follow the given bounding boxes order)}}
For the predicted action, choose the two pid that have highest probability of doing this action.
use json file.
'''

import os
from os.path import join
from os import mkdir, listdir

from collections import OrderedDict

import cv2
import PIL
import numpy as np
import torch

import xml.etree.ElementTree as ET

import random
from PIL import Image

import numpy as np

from collections import Counter
import os

from pathlib import Path

from os.path import exists

from copy import deepcopy
import json
import shutil

import tqdm

from config import *
from distant import distant_read_dataset
from utils import print_log
from distant_utils import readDict

from distant import ACTIONS, ACTIVITIES

from sklearn.metrics import precision_recall_fscore_support


def saveDict(src: dict, save_path: str):
    # Notice that, here the dict is not OrderDict
    with open(save_path, 'w') as f:
        json.dump(src, f)


def safeMkdir(path: str):
    if exists(path):
        return
    else:
        mkdir(path)


def drawBoundingBox(imgPath: str, bboxes: dict, savePath: str):
    img = cv2.imread(imgPath)
    img = cv2.UMat(img).get()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for key in bboxes.keys():
        bbox = bboxes[key]
        bbox = [int(i) for i in bbox]
        text = str(key)
        f = lambda x: (int(x[0]), int(x[1]))
        cv2.rectangle(img, f(bbox[:2]), f(bbox[2:]), (0, 255, 0), 4)
        cv2.putText(img, text, f(bbox[:2]), font, 2, (0, 0, 255), 1)
    cv2.imwrite(savePath, img)


def plot_confusion_matrix(cm,
                          target_names,
                          save_path='./result.svg',
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          ):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    save_path:    path to save image.

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    plot_confusion_matrix(cm           = np.array([[ 1098,  1934,   807],
                                              [  604,  4392,  6233],
                                              [  162,  2362, 31760]]),
                      normalize    = False,
                      target_names = ['high', 'medium', 'low'],
                      title        = "Confusion Matrix")

    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    matplotlib.use('Agg')
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(save_path)
    plt.close()


def evaluateDistant(cfg, result, fast=False):
    # TODO: check conflicts
    AC_NUM = len(ACTIVITIES)
    # Confuse matrix
    interaction_num_matrix = np.zeros((AC_NUM, AC_NUM))

    each_interaction_true_positive_detect_p_num = [0 for i in ACTIVITIES]
    each_interaction_tp_fn_fp_detect_p_num = [0 for i in ACTIVITIES]

    each_interaction_ip = [0 for i in ACTIVITIES]
    each_interaction_num = [0 for i in ACTIVITIES]

    interaction_preds = []
    interaction_gts = []

    action_preds = []
    action_gts = []
    detection_failed_nums = []

    actors_preds = []
    actors_gts = []

    each_actor_preds = {}
    each_actors_gts = {}

    seqs = result['interactions']['prediction'].keys()

    action_labels = [i for i in range(1, AC_NUM + 1)]  # Here, we exclude the negative class 0.
    interaction_labels = [i for i in range(AC_NUM)]

    for interaction in interaction_labels:
        each_actor_preds[interaction] = []
        each_actors_gts[interaction] = []

    mst = 0
    fpt = 0
    fct = 0
    dt = 0
    gt = 0

    for seq in seqs:
        detection_failed_nums += torch.cat(result['detection_failed_num'][seq])
        interaction_preds.append(result['interactions']['prediction'][seq])
        interaction_gts.append(result['interactions']['gt'][seq])
        action_preds += torch.cat(result['actions']['prediction'][seq]).flatten()
        action_gts += torch.cat(result['actions']['gt'][seq]).flatten()
        actor_p_temp = torch.cat(result['actors']['prediction'][seq]).flatten()
        
        # since there exists detection failed boxes, which has already be considered in action metrics but not in actor metrics, and I still want to use the sklearn to calculate it. I will add 1 into actor gts according to detection_failed_nums and consider the true negative place to replace.
        c_detection_failed_num=torch.cat(result['detection_failed_num'][seq]).flatten() # list, len:f_num
        c_actor_gt=torch.cat(result['actors']['gt'][seq])
        c_actor_p=torch.cat(result['actors']['prediction'][seq])
        f_num,p_num=c_actor_gt.shape
        for f_i in range(f_num):
            if c_detection_failed_num[f_i]>0:
                num_add=c_detection_failed_num[f_i]
                point=0
                while(num_add>0 and point<p_num):
                    if c_actor_gt[f_i,point]==0 and c_actor_p[f_i,point]==0:
                        c_actor_gt[f_i,point]=1
                        num_add-=1
                    point+=1
                if point==p_num:
                    print('ERROR: no place for add detection failures, there must be something wrong with detection_failed_nums.')
                    exit(0)
        
        actor_gt_temp = c_actor_gt.flatten()
        actors_preds += actor_p_temp
        actors_gts += actor_gt_temp
        each_actor_preds[int(result['interactions']['gt'][seq].item())].append(
            torch.cat(result['actors']['prediction'][seq]).flatten())
        each_actors_gts[int(result['interactions']['gt'][seq].item())].append(
            c_actor_gt.flatten())
        if result['interactions']['prediction'][seq] != result['interactions']['gt'][seq]:
            fct += ((actor_p_temp == actor_gt_temp) & (actor_gt_temp == 1)).sum()

    interaction_preds = torch.Tensor(interaction_preds)  # S
    interaction_gts = torch.Tensor(interaction_gts)  # S

    action_preds = torch.Tensor(action_preds)  # S*T*N
    action_gts = torch.Tensor(action_gts)  # S*T*N

    actors_preds = torch.Tensor(actors_preds)
    actors_gts = torch.Tensor(actors_gts)

    detection_failed_nums = torch.Tensor(detection_failed_nums)

    mst = detection_failed_nums.sum() + ((actors_preds == 0) & (actors_gts == 1)).sum()
    fpt = ((actors_preds == 1) & (actors_gts == 0)).sum()
    MHIA = 1 - (mst + fpt + fct) / (actors_preds.sum() + (actors_gts.sum() + detection_failed_nums.sum()))

    each_interaction_prec, each_interaction_recall, each_interaction_f1, _ = precision_recall_fscore_support(
        interaction_gts.numpy(), interaction_preds.numpy(), average=None, labels=interaction_labels)

    actions_TP = ((action_preds == action_gts) & (action_gts > 0)).sum()
    actions_FP = ((action_preds > 0) & (action_preds != action_gts)).sum()
    actions_FN = ((action_preds != action_gts) & (action_gts > 0)).sum() + detection_failed_nums.sum()
    actions_recall = actions_TP / (actions_TP + actions_FN)
    actions_precision = actions_TP / (actions_TP + actions_FP)
    actions_f1 = 2 * actions_precision * actions_recall / (actions_precision + actions_recall)

    # each_action_prec,each_action_recall,each_action_f1,_=precision_recall_fscore_support(action_gts.numpy(), action_preds.numpy(), average=None,labels=action_labels)

    actor_prec, actor_recall, actor_f1, _ = precision_recall_fscore_support(actors_gts.numpy(), actors_preds.numpy(),
                                                                            average='binary')

    each_actor_prec, each_actor_recall, each_actor_f1 = [], [], []
    for interaction in each_actor_preds.keys():
        p, r, f, _ = precision_recall_fscore_support(torch.cat(each_actors_gts[interaction]).numpy(),
                                                     torch.cat(each_actor_preds[interaction]).numpy(), average='binary')
        each_actor_prec.append(p)
        each_actor_recall.append(r)
        each_actor_f1.append(f)

    report = {}
    report['MHIA'] = MHIA.tolist()
    report['interaction_f1'] = each_interaction_f1.mean().tolist()
    report['interaction_prec'] = each_interaction_prec.mean().tolist()
    report['interaction_recall'] = each_interaction_recall.mean().tolist()
    report['each_interaction_f1'] = each_interaction_f1.tolist()
    report['each_interaction_prec'] = each_interaction_prec.tolist()
    report['each_interaction_recall'] = each_interaction_recall.tolist()
    report['perf_action_f1'] = actions_f1.tolist()
    report['perf_action_prec'] = actions_precision.tolist()
    report['perf_action_recall'] = actions_recall.tolist()
    # report['perf_each_action_f1']=each_action_f1.tolist()
    # report['perf_each_action_prec']=each_action_prec.tolist()
    # report['perf_each_action_recall']=each_action_recall.tolist()
    report['actor_f1'] = np.array(each_actor_f1).mean().tolist()
    report['actor_prec'] = np.array(each_actor_prec).mean().tolist()
    report['actor_recall'] = np.array(each_actor_recall).mean().tolist()
    report['each_actor_f1'] = each_actor_f1
    report['each_actor_prec'] = each_actor_prec
    report['each_actor_recall'] = each_actor_recall
    report['interaction_acc'] = (interaction_gts == interaction_preds).float().mean().tolist()

    safeMkdir(cfg.result_path)
    print_log(cfg.log_path, report)
    saveDict(report, join(cfg.result_path, 'report.json'))
    interaction_labels=[1,2,3,0,5,4]
    action_labels=[0,2,3,4,1,6,5]
    for interaction_a in interaction_labels:
        for interaction_b in interaction_labels:
            interaction_num_matrix[interaction_labels.index(interaction_a), interaction_labels.index(interaction_b)] = (
                        (interaction_gts == interaction_a) & (interaction_preds == interaction_b)).sum()

    if not fast:
        a = torch.zeros(7, 7)
        for _i in action_labels:
            for _j in action_labels:
                a[action_labels.index(_i),action_labels.index(_j)] = ((action_preds == _j) & (action_gts == _i)).sum()
        plot_confusion_matrix(a.int().numpy(), save_path=join(cfg.result_path, 'action.svg'), normalize=True,
                              target_names=['NA', 'CO', 'GO', 'GR', 'CH', 'TC', 'PP'],
                              title='Confustion Matrix')
        plot_confusion_matrix(interaction_num_matrix, save_path=join(cfg.result_path, 'interaction.svg'),
                              normalize=True,
                              target_names=['CO', 'GO', 'GR', 'CH', 'TC', 'PP'], title='Confusion Matrix')
    return report['MHIA']


if __name__ == '__main__':
    cfg = Config('Distant')

    cfg.device_list = "0,1"
    cfg.use_multi_gpu = True
    cfg.training_stage = 2
    cfg.stage1_model_path = 'result/STAGE1_MODEL.pth'  # PATH OF THE BASE MODEL
    cfg.train_backbone = False
    cfg.test_before_train = True

    cfg.image_size = 480, 720
    cfg.out_size = 57, 87
    cfg.num_boxes = 13
    cfg.num_actions = 7
    cfg.num_activities = 6
    cfg.num_frames = 10
    cfg.num_graph = 4
    cfg.tau_sqrt = True

    cfg.batch_size = 32
    cfg.test_batch_size = 8
    cfg.train_learning_rate = 1e-4
    cfg.train_dropout_prob = 0.2
    cfg.weight_decay = 1e-2
    cfg.lr_plan = {}
    cfg.max_epoch = 50

    cfg.exp_note = 'Distant_stage2'
    test_anns = distant_read_dataset(join(cfg.data_path, 'annotations'), cfg)
    # print(test_anns)
    # plot_confusion_matrix(cm=np.array([[1098, 1934, 807],
    #                                    [604, 4392, 6233],
    #                                    [162, 2362, 31760]]),
    #                       save_path='./test.svg',
    #                       normalize=False,
    #                       target_names=['high', 'medium', 'low'],
    #                       title="Confusion Matrix")
    result = readDict(
        '/home/molijuly/myProjects/HIT/result/[Distant_stage2_stage2]<2020-10-30_01-57-01>/result_epoch0.json',
        key_is_int=True)
    evaluateDistant(cfg, result, fast=False)
