import os
import numpy as np
#
# path and dataset parameter
#

DATA_PATH = 'data'
PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')
CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')#数据文件夹


OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')#输出文件夹
WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')#权重文件夹

WEIGHTS_FILE = None
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')


def read_coco_labels():
    f = open("data/coco_classes.txt")
    class_names = []
    for l in f.readlines():
        l = l.strip() # 去掉回车'\n'
        class_names.append(l)
    return class_names

CLASSES_NAMES = read_coco_labels()
CLASSES = np.array(range(len(CLASSES_NAMES)))
FLIPPED = True


#
# model parameter
#
ANCHORS = [[0.57273, 0.677385],
           [1.87446, 2.06253],
           [3.33843, 5.47434],
           [7.88282, 3.52778],
           [9.77052, 9.16828]]

IMAGE_SIZE = 416#图片大小(416,416)

CELL_SIZE = 13#划分格子7*7

# BOXES_PER_CELL = 2#每个格子两个box

ALPHA = 0.1#激活函数 系数

DISP_CONSOLE = False##控制台输出信息
#四个损失函数系数
OBJECT_SCALE = 1.0#有目标时，置信度权重
NOOBJECT_SCALE = 1.0#没有目标时，置信度权重
CLASS_SCALE = 2.0#类别权重
COORD_SCALE = 5.0 #边界框权重

N_LAST_CHANNELS = 425

#
# solver parameter
#

GPU = ''

LEARNING_RATE = 0.0001

DECAY_STEPS = 30000#学习率衰减步数

DECAY_RATE = 0.1#衰减率

STAIRCASE = True

BATCH_SIZE = 45

MAX_ITER = 15000

SUMMARY_ITER = 10#日志文件保存间隔步

SAVE_ITER = 1000




#
# test parameter
#

THRESHOLD = 0.2#格子有目标的置信度阈值

NMS_THRESHOLD = 0.5#非极大值抑制 IoU阈值
