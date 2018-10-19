import random
import colorsys
import cv2
import numpy as np
import config as cfg

def preprocess_image(image,image_size=cfg.IMAGE_SIZE):
    """
    预处理图片,resize and normalize，加上一个维度
    args:
        image:输入图片
        image_size:resize dax
    return
        image_expanded:返回图片
    """
    image_cp = np.copy(image).astype(np.float32)# 复制原图像

    # resize image
    image_rgb = cv2.cvtColor(image_cp,cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb,image_size)

    # normalize归一化
    image_normalized = image_resized.astype(np.float32) / 225.0

    # 增加一个维度在第0维——batch_size
    image_expanded = np.expand_dims(image_normalized,axis=0)

    return image_expanded