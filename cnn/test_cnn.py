#coding=utf-8

import os,glob
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
# 数据文件夹
train_dir = "data/train"
# 训练还是测
train = 0
# 模型文件路径
model_path = "model/image_model"
label_name_dict = {0:'9', 1:'0', 2:'7',3: '+',
                           4:'6',5: '1',6: '8', 7:'-',
                           8:'d',9: '=', 10:'4', 11:'x',
                           12:'3',13: '2',14: '5'}





def read_test(test_dir):
    datas=[]
    next_path = os.listdir(test_dir)
    for files in next_path:
        img = cv2.imread(files)
        #img_processed=pre_process(img)
        data = np.array(img)
        datas.append(data)
    datas = np.array(datas)

    return datas



def cnn_recognizer(img):
    # 定义Placeholder，存放输入和标签
    datas_placeholder = tf.placeholder(tf.float32, [None, 45,45, 3])
    labels_placeholder = tf.placeholder(tf.int32, [None])

    # 存放DropOut参数的容器，训练时为0.25，测试时为0
    dropout_placeholdr = tf.placeholder(tf.float32)

    # 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
    conv0 = tf.layers.conv2d(datas_placeholder, 32, 3, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool0 = tf.layers.max_pooling2d(conv0, [3, 3], [3, 3])

    # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
    conv1 = tf.layers.conv2d(pool0, 64, 3, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool1 = tf.layers.max_pooling2d(conv1, [3, 3], [3, 3])


    # # 将3维特征转换为1维向量
    flatten = tf.layers.flatten(pool1)

    # 全连接层，转换为长度为100的特征向量
    fc = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)

    # 加上DropOut，防止过拟合
    dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

    # 未激活的输出层
    logits = tf.layers.dense(dropout_fc, 15)

    predicted_labels = tf.arg_max(logits, 1)


    # 利用交叉熵定义损失
    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(labels_placeholder, 15),
        logits=logits
    )
    # 平均损失
    mean_loss = tf.reduce_mean(losses)

    # 定义优化器，指定要优化的损失函数
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(losses)


    # 用于保存和载入模型
    saver = tf.train.Saver()


    with tf.Session() as sess:
        test_single = []
        processed = cv2.resize(img, (45, 45))
        data = np.array(processed)
        test_single.append(data)

        print("测试模式")
        # 如果是测试，载入参数
        saver.restore(sess, model_path)
        print("从{}载入模型".format(model_path))
        # label和名称的对照关系
        label_name_dict = {0: '9', 1: '0', 2: '7', 3: '+',
                           4: '6', 5: '1', 6: '8', 7: '-',
                           8: 'd', 9: '=', 10: '4', 11: 'x',
                           12: '3', 13: '2', 14: '5'}
        # 定义输入和Label以填充容器，测试时dropout为0
        test_feed_dict = {
            datas_placeholder: test_single,
            dropout_placeholdr: 0
        }
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        # 真实label与模型预测label
        count = 0
        for  i in predicted_labels_val:
            # 将label id转换为label名
            predicted_label_name = label_name_dict[i]
            print("cnn recognizes as :",predicted_label_name)
            return predicted_label_name







