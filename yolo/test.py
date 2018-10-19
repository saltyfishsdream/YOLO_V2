import os
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.utils import cal_iou
import random
import colorsys
import cv2

class TEST(object):
    def __init__(self):
        self.alpha = cfg.ALPHA
        self.n_last_channels = cfg.N_LAST_CHANNELS
        self.cell_size = cfg.CELL_SIZE
        self.anchors = cfg.ANCHORS
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.threshold = cfg.THRESHOLD
        self.nms_threshold = cfg.NMS_THRESHOLD
    def decode(self,model_output):
        '''
        将测试时的网络中的数据解码,将位置信息变为（left_up,right_down）
        args:
            model_output:darknet19网络输出的特征图
        return:
            bboxs:解码后的bb框位置
            obj_probs：检测物体概率，即置信度
            class_probs：类别概率
        '''
        H = W = self.cell_size
        num_anchors = len(self.anchors) # 这里的anchor是在configs文件中设置的
        anchors = tf.constant(self.anchors, dtype=tf.float32)  # 将传入的anchors转变成tf格式的常量列表

        # 13*13*num_anchors*(num_class+5)，第一个维度自适应batchsize
        detection_result = tf.reshape(model_output,[-1,H*W,num_anchors,self.num_classes+5])

        # darknet19网络输出转化——偏移量、置信度、类别概率
        xy_offset = tf.nn.sigmoid(detection_result[:,:,:,0:2]) # 中心坐标相对于该cell左上角的偏移量，sigmoid函数归一化到0-1
        wh_offset = tf.exp(detection_result[:,:,:,2:4]) #相对于anchor的wh比例，通过e指数解码
        obj_probs = tf.nn.sigmoid(detection_result[:,:,:,4]) # 置信度，sigmoid函数归一化到0-1
        class_probs = tf.nn.softmax(detection_result[:,:,:,5:]) # 网络回归的是'得分',用softmax转变成类别概率

        # 构建特征图每个cell的左上角的xy坐标
        height_index = tf.range(H,dtype=tf.float32) # range(0,13)
        width_index = tf.range(W,dtype=tf.float32) # range(0,13)
        # 变成x_cell=[[0,1,...,12],...,[0,1,...,12]]和y_cell=[[0,0,...,0],[1,...,1]...,[12,...,12]]
        x_cell,y_cell = tf.meshgrid(height_index,width_index)
        x_cell = tf.reshape(x_cell,[1,-1,1]) # 和上面[H*W,num_anchors,num_class+5]对应
        y_cell = tf.reshape(y_cell,[1,-1,1])

        # decode
        bbox_x = (x_cell + xy_offset[:,:,:,0]) / W
        bbox_y = (y_cell + xy_offset[:,:,:,1]) / H
        bbox_w = (anchors[:,0] * wh_offset[:,:,:,0]) / W
        bbox_h = (anchors[:,1] * wh_offset[:,:,:,1]) / H
        # 中心坐标+宽高box(x,y,w,h) -> xmin=x-w/2 -> 左上+右下box(xmin,ymin,xmax,ymax)
        bboxes = tf.stack([bbox_x-bbox_w/2, bbox_y-bbox_h/2,
                        bbox_x+bbox_w/2, bbox_y+bbox_h/2], axis=3)

        return bboxes, obj_probs, class_probs
    
    def postprocess(self,bboxes,obj_probs,class_probs):
        """
        对解码后输出结果的处理
        args：
            bboxes：解码后的位置信息
            obj_probs:解码后的置信度
            class_probs:解码后的类别信息
        return：
            bboxes：nms后的位置信息
            obj_probs:nms后的置信度
            class_max_index:nms后的类别index
        """
        bboxes = np.reshape(bboxes,[-1,4])#4列分别是box(xmin,ymin,xmax,ymax)
        
        # 将所有box还原成图片中真实的位置
        bboxes[:,0:1] *= float(self.image_size) # xmin*width
        bboxes[:,1:2] *= float(self.image_size) # ymin*height
        bboxes[:,2:3] *= float(self.image_size) # xmax*width
        bboxes[:,3:4] *= float(self.image_size) # ymax*height

        # 置信度*max类别概率=类别置信度scores
        obj_probs = np.reshape(obj_probs,[-1])
        class_probs = np.reshape(class_probs,[len(obj_probs),-1])
        class_max_index = np.argmax(class_probs,axis=1)
        class_probs = class_probs[np.arange(len(obj_probs)),class_max_index]
        scores = obj_probs * class_probs

        # 类别置信度scores>threshold的边界框bboxes留下
        keep_index = scores > self.threshold
        class_max_index = class_max_index[keep_index]
        scores = scores[keep_index]
        bboxes = bboxes[keep_index]

        #NMS
        class_max_index,scores,bboxes = self.bboxes_nms(class_max_index,scores,bboxes)

        return bboxes,scores,class_max_index

        
    def bboxes_nms(self,cl_max,scores,bboxes,top_k=-1):
        """
        执行NMS操作
        args:
            scores:置信度
            bboxs：选框
        returns:
            bboxes[idxes]：nms后的位置信息
            scores[idxes]:nms后的置信度
            classes[idxes]:nms后的类别index
        """
        # 排序并取前top_k
        index = np.argsort(-scores)
        # top_k = 10
        cl_max = cl_max[index][:top_k]
        scores = scores[index][:top_k]
        bboxes = bboxes[index][:top_k]

        #NMS
        keep_bboxes = np.ones(scores.shape, dtype=np.bool)
        for i in range(scores.size-1):
            if keep_bboxes[i]:
                # Computer overlap with bboxes which are following.
                overlap = cal_iou(bboxes[i], bboxes[(i+1):])
                # Overlap threshold for keeping + checking part of the same class
                keep_overlap = np.logical_or(overlap < self.nms_threshold, cl_max[(i+1):] != cl_max[i])
                keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

        idxes = np.where(keep_bboxes)
        return cl_max[idxes], scores[idxes], bboxes[idxes]

    def draw_detection(self,im, bboxes, scores, cls_inds, labels):
        """
        绘制框
        args:
            im:原图片
            bboxs：选框
            scores：置信度
            cls_inds：类别索引
            labels：类别名称集合
        returns:
            imgcv：绘制框后图片
        """
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x/float(len(labels)), 1., 1.)  for x in range(len(labels))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        # draw image
        imgcv = np.copy(im)
        h, w, _ = imgcv.shape
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]

            thick = int((h + w) / 300)
            cv2.rectangle(imgcv,(box[0], box[1]), (box[2], box[3]),colors[cls_indx], thick)
            mess = '%s: %.3f' % (labels[cls_indx], scores[i])
            if box[1] < 20:
                text_loc = (box[0] + 2, box[1] + 15)
            else:
                text_loc = (box[0], box[1] - 10)
            # cv2.rectangle(imgcv, (box[0], box[1]-20), ((box[0]+box[2])//3+120, box[1]-8), (125, 125, 125), -1)  # puttext函数的背景

            cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3*h, (255,255,255), thick//3)
        return imgcv
