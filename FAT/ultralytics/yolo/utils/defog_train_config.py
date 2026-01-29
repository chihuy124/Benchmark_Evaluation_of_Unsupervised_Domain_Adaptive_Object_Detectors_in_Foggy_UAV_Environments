import argparse
from easydict import EasyDict as edict
from ultralytics.yolo.utils.filters import *
MODEL = 'PSPNet'  # PSPNet, DeepLab, RefineNet

BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 0

SET = 'train'

INPUT_SIZE = '512'
INPUT_SIZE_TARGET = '960'

NUM_CLASSES = 19
IGNORE_LABEL = 255

LEARNING_RATE = 2.5e-4
POWER = 0.9
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LEARNING_RATE_D = 1e-4

NUM_STEPS = 50000
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'+MODEL
STD = 0.05

__C                             = edict()


cfg                             = __C


cfg.filters = [DefogFilter]


cfg.num_filter_parameters = 15

cfg.defog_begin_param = 0

cfg.wb_begin_param = 1
cfg.gamma_begin_param = 4
cfg.tone_begin_param = 5
cfg.contrast_begin_param = 13
cfg.usm_begin_param = 14


#
cfg.exposure_begin_param = 0



cfg.defog_range = (0.0, 1.0)
