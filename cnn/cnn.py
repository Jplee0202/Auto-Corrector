#coding=utf-8

import os,glob
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
from identify import pre_process
# 数据文件夹
train_dir = "data/train"
test_dir="data/validation"
# 训练还是测
train = 0
# 模型文件路径
model_path = "model/image_model"

label_name_dict = {0:'9', 1:'0', 2:'7',3: '+',
                           4:'6',5: '1',6: '8', 7:'-',
                           8:'d',9: '=', 10:'4', 11:'x',
                           12:'3',13: '2',14: '5'}


label_name_dict2 = {0: '+', 1: '-', 2: 'd', 3: '=',
                           4: 'x'}

set_train = {}
if os.path.isdir(train_dir):
    next_path = os.listdir(train_dir)
    # print(next_path)
    for files in next_path:
        if str(files) == ".DS_Store":
            continue
        file = train_dir + "/" + str(files)
        n_file = glob.glob(file + "/*.jpg")
        set_train[files] = len(n_file)


set_test = {}
if os.path.isdir(test_dir):
    next_path = os.listdir(test_dir)
    # print(next_path)
    for files in next_path:
        if str(files) == ".DS_Store":
            continue
        file = test_dir + "/" + str(files)
        n_file = glob.glob(file + "/*.jpg")
        set_test[files] = len(n_file)


def read_train_data(data_dir):
    datas = []
    labels = []
    dict_id=0
    for folder, number in set_train.items():
        dire = train_dir + "/" + folder + "/"
        print("Train: ", str(dict_id + 1))
        for i in range(number):
            file_name = dire + folder + "_" + str(i + 1) + ".jpg"
            print(file_name)
            img = cv2.imread(file_name)
            data = np.array(img)
            label=dict_id
            datas.append(data)
            labels.append(label)
        dict_id+=1
    datas = np.array(datas)
    labels = np.array(labels)

    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return datas, labels


def read_test_data(data_dir):
    datas = []
    labels = []
    dict_id=0
    for folder, number in set_test.items():
        dire = test_dir + "/" + folder + "/"
        print("Test: ", str(dict_id + 1))
        for i in range(number):
            file_name = dire + folder + "_" + str(i + 1) + ".jpg"
            print(file_name)
            img = cv2.imread(file_name)
            #img_processed=pre_process(img)
            data = np.array(img)
            label=dict_id
            datas.append(data)
            labels.append(label)
        dict_id+=1
    datas = np.array(datas)
    labels = np.array(labels)

    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return datas, labels

datas, labels = read_train_data(train_dir)
test_data,test_labels=read_test_data(test_dir)


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

    if train:
        print("训练模式")
        # 如果是训练，初始化参数
        sess.run(tf.global_variables_initializer())
        # 定义输入和Label以填充容器，训练时dropout为0.25
        train_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0.6
        }
        for step in range(70):
            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)

            if step % 10 == 0:
                print("step = {}\tmean loss = {}".format(step, mean_loss_val))
        saver.save(sess, model_path)
        print("训练结束，保存模型到{}".format(model_path))

    else:
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
            datas_placeholder: test_data,
            dropout_placeholdr: 0
        }
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        # 真实label与模型预测label
        count = 0
        for  real_label, predicted_label in zip( test_labels, predicted_labels_val):
            # 将label id转换为label名
            real_label_name = label_name_dict[real_label]
            predicted_label_name = label_name_dict[predicted_label]
            print("real: ",real_label_name,"predict: ",predicted_label_name)








