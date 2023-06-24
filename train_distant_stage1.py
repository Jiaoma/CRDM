from train_net import *
from distant import preprocess
import sys
sys.path.append(".")

cfg = Config('Distant')

cfg.num_workers=4
cfg.device_list = "0,"
cfg.use_multi_gpu = True
cfg.training_stage = 1
cfg.train_backbone = True
cfg.stage1_model_path='result/stage1_ac9.pth'
cfg.test_interval_epoch=10

cfg.image_size = 480, 720
cfg.out_size = 57, 87
cfg.num_boxes = 30
cfg.num_actions = 7
cfg.num_activities = 6
cfg.num_frames = 5
cfg.K=5

cfg.batch_size = 64
cfg.test_batch_size = 64
cfg.train_learning_rate = 1e-5
cfg.train_dropout_prob = 0.5
cfg.weight_decay = 1e-2
cfg.lr_plan = {}
cfg.max_epoch = 100
cfg.test_before_train=False

cfg.exp_note = 'Distant_stage1'
# preprocess(join(cfg.data_path, 'frames'), cfg.image_size)
train_net(cfg)
# test_net(cfg)