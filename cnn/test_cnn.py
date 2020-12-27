#coding=utf-8

import os,glob
from PIL import Image
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
from cnn import pre_processor
train_dir = "data/train"
train = 0
model_path = "cnn/model/image_model"
label_name_dict = {0:'9', 1:'0', 2:'7',3: '+',
                           4:'6',5: '1',6: '8', 7:'-',
                           8:'d',9: '=', 10:'4', 11:'x',
                           12:'3',13: '2',14: '5'}



#define placeholder
datas_placeholder = tf.placeholder(tf.float32, [None, 45,45, 3])
labels_placeholder = tf.placeholder(tf.int32, [None])
# save dropout train:0.6, test:0
dropout_placeholdr = tf.placeholder(tf.float32)

# 32*3*3
conv0 = tf.layers.conv2d(datas_placeholder, 32, 3, activation=tf.nn.relu)
#max-pooling 2*2 stride 2*2
pool0 = tf.layers.max_pooling2d(conv0, [3, 3], [3, 3])

#40*4*4
conv1 = tf.layers.conv2d(pool0, 64, 3, activation=tf.nn.relu)
#max-pooling 2*2 stride:2
pool1 = tf.layers.max_pooling2d(conv1, [3, 3], [3, 3])

#flaten 3d into 1d
flatten = tf.layers.flatten(pool1)

#fc, turn into 100d
fc = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)

# avoide overfitting
dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

logits = tf.layers.dense(dropout_fc, 15)

predicted_labels = tf.arg_max(logits, 1)


losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, 15),
    logits=logits
)
# average loss
mean_loss = tf.reduce_mean(losses)

# loss function
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(losses)

# save model
saver = tf.train.Saver()



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
    processed=pre_processor.pre_process(img)
    processed=cv2.resize(processed,(45,45))

    with tf.Session() as sess:
        test_single = []
        data = np.array(processed)
        test_single.append(data)

        saver.restore(sess, model_path)
        label_name_dict = {0: '9', 1: '0', 2: '7', 3: '+',
                           4: '6', 5: '1', 6: '8', 7: '-',
                           8: 'd', 9: '=', 10: '4', 11: 'x',
                           12: '3', 13: '2', 14: '5'}
        test_feed_dict = {
            datas_placeholder: test_single,
            dropout_placeholdr: 0
        }
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        count = 0
        for i in predicted_labels_val:
            # turn into corresponding label
            predicted_label_name = label_name_dict[i]
            return predicted_label_name






