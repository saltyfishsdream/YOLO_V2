import os
import numpy as np
import tensorflow as tf
import random
import colorsys
from yolo.net import YOLO
from yolo.test import TEST
from yolo.utils import preprocess_image
import yolo.config as cfg
import cv2

def detect(test,net,img,sess,model_output,tf_image):
    '''
        图片目标检测
        
        args:
            test:test类
            net:网络类
            img:原始图片数据
            sess:tf会话
        return：
            result：返回检测到的边界框，list类型 每一个元素对应一个目标框 
            包含{类别名,x_center,y_center,w,h,置信度}
        '''
    
    output_decoded = test.decode(model_output=model_output)

    #运行会话
    image_cp = preprocess_image(img)
    bboxes,obj_probs,class_probs = sess.run(output_decoded,feed_dict={tf_image:image_cp})
    bboxes,scores,class_max_index = test.postprocess(bboxes,obj_probs,class_probs)

    return bboxes,scores,class_max_index

def image_detector(sess,image_file, wait=0):
    '''
    目标检测
    
    args：
        image_file：测试图片路径
        sess:tf会话
    '''
    net = YOLO()
    test = TEST()
    #test
    tf_image = tf.placeholder(tf.float32,[1,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE,3])
    model_output = net.built_net(tf_image)
    saver = tf.train.Saver()
    saver.restore(sess, "data/yolo2_model/yolo2_coco.ckpt")

    image = cv2.imread(image_file)
    bboxes,scores,class_max_index = detect(test,net,image,sess,model_output,tf_image)
    dec_img = test.draw_detection(image, bboxes, scores, class_max_index, cfg.CLASSES)
    cv2.imshow('img',dec_img)
    cv2.waitKey(wait)

def main():
    sess = tf.Session()
    image_detector(sess,"data/a.jpg")
    sess = sess.close()

if __name__ == '__main__':
    # print(cfg.CLASSES)
    main()