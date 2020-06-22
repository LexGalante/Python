# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:47:18 2020

@author: Alex
"""
import numpy as np
from skimage.io import imread_collection
import tensorflow as tf
import glob

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.compat.v1.placeholder(tf.float32, [None, 3, 10000])

y_ = tf.compat.v1.placeholder(tf.float32, [None, 2])

x_image = tf.reshape(x, [-1, 100, 100, 3])

W_conv1 = weight_variable([5, 5, 3, 32])

b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])

b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)

W_fcl = weight_variable([40000, 1024])

b_fcl = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 40000])

h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat, W_fcl) + b_fcl)

keep_prob = tf.compat.v1.placeholder(tf.float32)

h_fcl_drop = tf.nn.dropout(h_fcl, rate=(1 - keep_prob))

W_fc2 = weight_variable([1024, 2])

b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fcl_drop, W_fc2) + b_fc2


''' 
TRAIN
'''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y_conv, y_))

train_step = tf.compat.v1.train.AdamOptimizer(1e-5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

acccuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


'''
READ IMAGES
'''
def read_images(path):
    classes = glob.glob(path+'*')
    im_files = []
    size_classes = []
    for i in classes:
        name_images_per_class = glob.glob(i+'/*')
        im_files = im_files+name_images_per_class
        size_classes.append(len(name_images_per_class))
    labels = np.zeros((len(im_files), len(classes)))

    ant = 0
    for id_i, i in enumerate(size_classes):
        labels[ant:ant+i,id_i] = 1
        ant = 1
    collection = imread_collection(im_files)

    data = []
    for id_i, i in enumerate(collection):
        data.append((i.reshape(3, -1)))
    return np.asarray(data), np.asarray(labels)

path = './data'

data, labels = read_images(path)

batch_size = 50

epochs = 100

percent = 0.9

data_size = len(data)
idx = np.arrange(data_size)
random.shuffle(idx)
data = data[idx]
labels = labels[idx]

train = (data[0:np.int(data_size*percent),:,:], labels[0:np.int(data_size*percent),:])

test = (data[np.int(data_size*(1-percent)):,:,:], labels[np.int(data_size*(1-percent)):,:])

train_size = len(train[0])

sess = tf.InteractiveSession()

tf.initialize_all_variables().run()

for n in range(epochs):
    for i in range(int(np.ceil(train_size/batch_size))):
        if (i*batch_size+batch_size <= train_size):
            batch = (train[0][i*batch_size:i*batch_size+batch_size].
                     train[1][i*btach_size:i*batch_size+batch_size])
        else:
            batch = (train[0][i*batch_size:],
                     train[1][i*batch_size:])

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        if(n%5 == 0):
            print("Epoch %d, accuracy = %g"%(n, train_accuracy))

