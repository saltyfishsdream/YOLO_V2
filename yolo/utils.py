import numpy as np
import yolo.config as cfg
import cv2

def cal_iou(bboxes1,bboxes2):
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
    int_h = np.maximum(int_ymax-int_ymin,0.)
    int_w = np.maximum(int_xmax-int_xmin,0.)

    # 计算IOU
    int_vol = int_h * int_w # 交集面积
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1]) # bboxes1面积
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1]) # bboxes2面积
    IOU = int_vol / (vol1 + vol2 - int_vol) # IOU=交集/并集
    return IOU
def preprocess_image(image,image_size=(cfg.IMAGE_SIZE,cfg.IMAGE_SIZE)):
    # 复制原图像
    image_cp = np.copy(image).astype(np.float32)

    # resize image
    image_rgb = cv2.cvtColor(image_cp,cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb,image_size)

    # normalize归一化
    # 没有 - 均值
    image_normalized = image_resized.astype(np.float32) / 225.0

    # 增加一个维度在第0维——batch_size
    image_expanded = np.expand_dims(image_normalized,axis=0)

    return image_expanded