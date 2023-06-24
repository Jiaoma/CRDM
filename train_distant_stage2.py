from train_net import *
from distant import preprocess
from train_net_social import train_net as train_net_social
import sys
sys.path.append(".")

cfg = Config('Distant')
cfg.num_workers=6
cfg.device_list = "0"
cfg.use_multi_gpu = True
cfg.training_stage = 2
# cfg.stage1_model_path = 'result/stage1_ac9.pth'  # PATH OF THE BASE MODEL
cfg.train_backbone = False
cfg.test_before_train = False
cfg.actions_loss_weight=1

cfg.image_size = 480, 720
cfg.out_size = 57,87 #184, 328
cfg.num_boxes = 30
cfg.num_actions = 7
cfg.num_activities = 6
cfg.tau_sqrt = True

cfg.num_features_boxes = 5
cfg.num_features_pose=128
cfg.num_features_mt=128
cfg.num_features_relation=256
cfg.num_features_gcn=128
cfg.num_features_relation_app=128
cfg.num_features_relation_mt=128

cfg.K=7
cfg.num_frames = cfg.K
cfg.continuous_frames=5
# cfg.interval=2

cfg.trace_features=5

cfg.batch_size = 16
cfg.test_batch_size = 16

cfg.train_learning_rate = 1e-5
cfg.train_dropout_prob = 0.2
cfg.weight_decay = 1e-2
cfg.lr_plan = {}
cfg.max_epoch = 100
cfg.stage2_model_path='/home/lijiacheng/code/HIT/result/[Distant_stage2_OneGroupV5Net_stage2]<2021-04-08_02-48-59>/stage2_epoch54_45.53%.pth'
# cfg.save_result_path='/home/lijiacheng/code/HIT/result/[Distant_stage2_OneGroupV5Net_K7_F3_stage2]<2021-04-06_10-58-39>/result_epoch24.pt'
cfg.ablation=False

cfg.exp_note = 'Distant_stage2_OneGroupV5Net+60'
# preprocess(join(cfg.data_path, 'frames'), cfg.image_size)
# For training, please uncomment the following line and comment test_net(cfg).
# train_net(cfg)
test_net(cfg)
