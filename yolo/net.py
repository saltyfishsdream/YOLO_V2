import os
import numpy as np
import tensorflow as tf
import yolo.config as cfg

class YOLO(object):
    def __init__(self):
        self.alpha = cfg.ALPHA
        self.n_last_channels = cfg.N_LAST_CHANNELS
        self.cell_size = cfg.CELL_SIZE
        self.anchors = cfg.ANCHORS
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
    
    ###################################
    "这个区间定义了网络的基本操作"
    
    def leaky_relu(self,x):
        "激活函数:leaky_relu"
        ""
        return tf.nn.leaky_relu(x,alpha=self.alpha,name='leaky_relu')
    
    def conv2d_bn(self,x,filters_num,filters_size,pad_size=0,stride=1,batch_normalize=True,activation=True,use_bias=False,name='conv2d'):
        """
        集合了卷积层和bn层操作
        args：
           x:输入数据
           filters_num:卷积核数目
           filters_size：卷积核大小
           pad_size：扩充尺寸
           stride：步幅
           batch_normalize：是否bn
           activation：激活函数
           use_bias：卷积层是否使用偏置
        return：
           out:返回卷积bn操作后的矩阵
        """
        if pad_size > 0:# padding，注意: 不用padding="SAME",否则可能会导致坐标计算错误
            x = tf.pad(x,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])

        # 有BN层，所以后面有BN层的conv就不用偏置bias，并先不经过激活函数activation
        out = tf.layers.conv2d(x,filters=filters_num,kernel_size=filters_size,strides=stride,
                                padding='VALID',activation=None,use_bias=use_bias,name=name)
        # BN，如果有，应该在卷积层conv和激活函数activation之间
        if batch_normalize:
            out = tf.layers.batch_normalization(out,axis=-1,momentum=0.9,training=False,name=name+'_bn')
        if activation:
            out = self.leaky_relu(out)
        return out

    def maxpool(self,x,size=2,stride=2,name='maxpool'):
        """
        最大池化
        args
            x:输入数据
            size：池化核大小
            stride：步幅
        """
        return tf.layers.max_pooling2d(x,pool_size=size,strides=stride)
    
    def reorg(self,x,stride):
        """
        passthrough层
        """
        return tf.space_to_depth(x,block_size=stride)
    ###################################


    ###################################
    "这个区间建立神经网络"
    def built_net(self,images):
        """
        建立darknet
        args:
            images:输入图像
            n_last_channels:最后一层输出的深度
        return：
            output:神经网络的输出结果
        """
        net = self.conv2d_bn(images, filters_num=32, filters_size=3, pad_size=1, name='conv1')
        net = self.maxpool(net, size=2, stride=2, name='pool1')

        net = self.conv2d_bn(net, 64, 3, 1, name='conv2')
        net = self.maxpool(net, 2, 2, name='pool2')

        net = self.conv2d_bn(net, 128, 3, 1, name='conv3_1')
        net = self.conv2d_bn(net, 64, 1, 0, name='conv3_2')
        net = self.conv2d_bn(net, 128, 3, 1, name='conv3_3')
        net = self.maxpool(net, 2, 2, name='pool3')

        net = self.conv2d_bn(net, 256, 3, 1, name='conv4_1')
        net = self.conv2d_bn(net, 128, 1, 0, name='conv4_2')
        net = self.conv2d_bn(net, 256, 3, 1, name='conv4_3')
        net = self.maxpool(net, 2, 2, name='pool4')

        net = self.conv2d_bn(net, 512, 3, 1, name='conv5_1')
        net = self.conv2d_bn(net, 256, 1, 0,name='conv5_2')
        net = self.conv2d_bn(net,512, 3, 1, name='conv5_3')
        net = self.conv2d_bn(net, 256, 1, 0, name='conv5_4')
        net = self.conv2d_bn(net, 512, 3, 1, name='conv5_5')
        shortcut = net # 存储这一层特征图，以便后面passthrough层
        net = self.maxpool(net, 2, 2, name='pool5')

        net = self.conv2d_bn(net, 1024, 3, 1, name='conv6_1')
        net = self.conv2d_bn(net, 512, 1, 0, name='conv6_2')
        net = self.conv2d_bn(net, 1024, 3, 1, name='conv6_3')
        net = self.conv2d_bn(net, 512, 1, 0, name='conv6_4')
        net = self.conv2d_bn(net, 1024, 3, 1, name='conv6_5')

        net = self.conv2d_bn(net, 1024, 3, 1, name='conv7_1')
        net = self.conv2d_bn(net, 1024, 3, 1, name='conv7_2')
        # shortcut增加了一个中间卷积层，先采用64个1*1卷积核进行卷积，然后再进行passthrough处理
        # 这样26*26*512 -> 26*26*64 -> 13*13*256的特征图
        shortcut = self.conv2d_bn(shortcut, 64, 1, 0, name='conv_shortcut')
        shortcut = self.reorg(shortcut, 2)
        net = tf.concat([shortcut, net], axis=-1) # channel整合到一起
        net = self.conv2d_bn(net, 1024, 3, 1, name='conv8')

        # detection layer:最后用一个1*1卷积去调整channel，该层没有BN层和激活函数
        output = self.conv2d_bn(net, filters_num=self.n_last_channels, filters_size=1, batch_normalize=False,
                        activation=None, use_bias=True, name='conv_dec')

        return output
    #######################################

    #######################################
    "用于训练"
    def loss_layer(self,predictions,targets,scales):
        """
        计算损失函数
        args:
            predictions:网络训练出的结果
            targets：目标标签
        return:
            loss:损失
        """
        pass
    #######################################
