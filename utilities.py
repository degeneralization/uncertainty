import tensorflow as tf
import numpy as np
import os
import scipy.io
import random
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import cv2

def crop(x):
  for q in range(len(x)):
    img = x[q]
    l = []
    for i in range(8):
      for j in range(8):
        im = img[i*4:(i+1)*4,j*4:(j+1)*4,:]
        l.append(im)
     
    l1 = np.random.permutation(l)
    
    t=0
    for i in range(8):
      for j in range(8):
        img[i*4:(i+1)*4,j*4:(j+1)*4,:] = l1[t]
        t+=1
    x[q] = img
  return x

def crop1(x):
  for q in range(len(x)):
    img = x[q]
    l = []
    for i in range(4):
      for j in range(4):
        im = img[i*8:(i+1)*8,j*8:(j+1)*8,:]
        l.append(im)
     
    l1 = np.random.permutation(l)
    
    t=0
    for i in range(4):
      for j in range(4):
        img[i*8:(i+1)*8,j*8:(j+1)*8,:] = l1[t]
        t+=1
    x[q] = img
  return x

def crop_for_mnist(x):
  for q in range(len(x)):
    img = x[q]
    l = []
    for i in range(7):
      for j in range(7):
        im = img[i*4:(i+1)*4,j*4:(j+1)*4,:]
        l.append(im)
     
    l1 = np.random.permutation(l)
    
    t=0
    for i in range(7):
      for j in range(7):
        img[i*4:(i+1)*4,j*4:(j+1)*4,:] = l1[t]
        t+=1
    x[q] = img
  return x

def pad3D(c_x, padlen=1):
    batch,m,n,r = c_x.shape
    c_y = np.zeros((batch,m+2*padlen, n+2*padlen, r),dtype=c_x.dtype)
    c_y[:, padlen:-padlen, padlen:-padlen,:] = c_x
    return c_y

def randomCrop_helper(img, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width,:]
    return img

def randomCrop(img, width, height):
    images = []
    for i in range(len(img)):
        tmp=randomCrop_helper(img[i], width, height)
        images.append(tmp)
    images = np.array(images)
    return images
  
def stand0(im):
  im = im.astype(np.float64,copy=False)
  mean = np.mean(im)
  std = np.std(im)
  std1 = max(std,1./np.sqrt(np.array(im.size,dtype = np.float64)))
  im = (im-mean)/std1
  return im

def stand(images):
  for i in range(len(images)):
    images[i] = stand0(images[i])
  return images 

def stand10(safe_images):
    safe_images = (safe_images - np.min(safe_images))/(np.max(safe_images) - np.min(safe_images))
    return safe_images
  
def stand1(images):
  for i in range(len(images)):
    images[i] = stand10(images[i])
  return images 

def flipone(img):
    a=np.random.randint(1,3)
    if a == 1:
        return img[:,::-1,...]
    else:
        return img

def randomflipbatch(batch):
    for i in range(len(batch)):
        batch[i] = flipone(batch[i])
    return batch

def preprocess(x_batch):
    x_batch = stand(x_batch)
    x_batch = pad3D(x_batch, padlen=4)
    x_batch = randomCrop(x_batch, 32, 32)
    x_batch = randomflipbatch(x_batch)
    return x_batch

def preprocess_for_mnist(x_batch):
    x_batch = stand(x_batch)
    return x_batch

def ErrorRateAt95Recall1(labels, scores):
    recall_point = 0.95
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    indices = np.argsort(scores)[::-1]    
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    n_match = sum(sorted_labels)
    n_thresh = recall_point * n_match
    thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
    FP = np.sum(sorted_labels[:thresh_index] == 0)
    TN = np.sum(sorted_labels[thresh_index:] == 0)
    return float(FP) / float(FP + TN)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

epsilon = 1e-3
def batch_norm(inputs, scale, beta, pop_mean, pop_var,training, decay = 0.99):
    batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
    train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
    if training:
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
    
def vgg16(network_name,dd,XX_list,YY_list,weight_decay): 
    with tf.variable_scope(network_name,reuse=tf.AUTO_REUSE):
        x = XX_list[network_name]
        y = YY_list[network_name]
        pop_var1 = tf.get_variable(network_name + "pop_var1", [64], trainable=False,initializer=tf.ones_initializer())
        pop_mean1 = tf.get_variable(network_name + "pop_mean1",[64], trainable=False,initializer=tf.zeros_initializer() )
        pop_var2 = tf.get_variable(network_name + "pop_var2",[64], trainable=False,initializer=tf.ones_initializer())
        pop_mean2 = tf.get_variable(network_name + "pop_mean2",[64], trainable=False,initializer=tf.zeros_initializer() )
        pop_var3 = tf.get_variable(network_name + "pop_var3",[128], trainable=False,initializer=tf.ones_initializer() )
        pop_mean3 = tf.get_variable(network_name + "pop_mean3",[128], trainable=False,initializer=tf.zeros_initializer() )
        pop_var4 = tf.get_variable(network_name + "pop_var4",[128], trainable=False,initializer=tf.ones_initializer())
        pop_mean4 = tf.get_variable(network_name + "pop_mean4",[128], trainable=False,initializer=tf.zeros_initializer())
        pop_var5 = tf.get_variable(network_name + "pop_var5",[256], trainable=False,initializer=tf.ones_initializer() )
        pop_mean5 = tf.get_variable(network_name + "pop_mean5",[256], trainable=False,initializer=tf.zeros_initializer())
        pop_var6 = tf.get_variable(network_name + "pop_var6",[256], trainable=False,initializer=tf.ones_initializer())
        pop_mean6 = tf.get_variable(network_name + "pop_mean6",[256], trainable=False,initializer=tf.zeros_initializer())
        pop_var7 = tf.get_variable(network_name + "pop_var7",[256], trainable=False,initializer=tf.ones_initializer() )
        pop_mean7 = tf.get_variable(network_name + "pop_mean7",[256], trainable=False,initializer=tf.zeros_initializer())
        pop_var8 = tf.get_variable(network_name + "pop_var8",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean8 = tf.get_variable(network_name + "pop_mean8",[512], trainable=False,initializer=tf.zeros_initializer() )
        pop_var9 = tf.get_variable(network_name + "pop_var9",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean9 = tf.get_variable(network_name + "pop_mean9",[512], trainable=False,initializer=tf.zeros_initializer())
        pop_var10 = tf.get_variable(network_name + "pop_var10",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean10 =tf.get_variable( network_name + "pop_mean10",[512], trainable=False,initializer=tf.zeros_initializer())
        pop_var11 = tf.get_variable( network_name + "pop_var11",[512], trainable=False,initializer=tf.ones_initializer())
        pop_mean11 = tf.get_variable(network_name + "pop_mean11",[512], trainable=False,initializer=tf.zeros_initializer() )
        pop_var12 = tf.get_variable( network_name + "pop_var12",[512], trainable=False,initializer=tf.ones_initializer())
        pop_mean12 = tf.get_variable(network_name + "pop_mean12",[512], trainable=False,initializer=tf.zeros_initializer() )
        pop_var13 = tf.get_variable(network_name + "pop_var13",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean13 = tf.get_variable(network_name + "pop_mean13",[512], trainable=False,initializer=tf.zeros_initializer() )

        scale1 = tf.get_variable(network_name + "scale1",[64],initializer=tf.ones_initializer())
        beta1 = tf.get_variable(network_name + "beta1",[64],initializer=tf.zeros_initializer())
        scale2 = tf.get_variable(network_name + "scale2",[64],initializer=tf.ones_initializer())
        beta2 = tf.get_variable(network_name + "beta2",[64],initializer=tf.zeros_initializer())
        scale3 = tf.get_variable(network_name + "scale3",[128],initializer=tf.ones_initializer() )
        beta3 = tf.get_variable(network_name + "beta3",[128],initializer=tf.zeros_initializer() )
        scale4 = tf.get_variable(network_name + "scale4",[128],initializer=tf.ones_initializer() )
        beta4 = tf.get_variable(network_name + "beta4",[128],initializer=tf.zeros_initializer() )
        scale5 = tf.get_variable( network_name + "scale5",[256],initializer=tf.ones_initializer())
        beta5 = tf.get_variable(network_name + "beta5",[256],initializer=tf.zeros_initializer())
        scale6 = tf.get_variable(network_name + "scale6",[256],initializer=tf.ones_initializer() )
        beta6 = tf.get_variable(network_name + "beta6",[256],initializer=tf.zeros_initializer() )
        scale7 = tf.get_variable(network_name + "scale7",[256],initializer=tf.ones_initializer() )
        beta7 = tf.get_variable(network_name + "beta7",[256],initializer=tf.zeros_initializer() )
        scale8 = tf.get_variable(network_name + "scale8",[512],initializer=tf.ones_initializer())
        beta8 = tf.get_variable(network_name + "beta8",[512],initializer=tf.zeros_initializer() )
        scale9 = tf.get_variable(network_name + "scale9",[512],initializer=tf.ones_initializer())
        beta9 = tf.get_variable(network_name + "beta9",[512],initializer=tf.zeros_initializer() )
        scale10 = tf.get_variable(network_name + "scale10",[512],initializer=tf.ones_initializer())
        beta10 = tf.get_variable(network_name + "beta10",[512],initializer=tf.zeros_initializer() )
        scale11 = tf.get_variable(network_name + "scale11",[512],initializer=tf.ones_initializer() )
        beta11 = tf.get_variable(network_name + "beta11",[512],initializer=tf.zeros_initializer())
        scale12 = tf.get_variable(network_name + "scale12",[512],initializer=tf.ones_initializer())
        beta12 = tf.get_variable(network_name + "beta12",[512],initializer=tf.zeros_initializer() )
        scale13 = tf.get_variable( network_name + "scale13",[512],initializer=tf.ones_initializer())
        beta13 = tf.get_variable(network_name + "beta13",[512],initializer=tf.zeros_initializer() )

        w_conv1 = tf.get_variable(network_name + 'w_conv1', [3,3,3,64], initializer = tf.contrib.layers.xavier_initializer())
        w_conv2 = tf.get_variable(network_name + 'w_conv2', [3,3,64,64], initializer = tf.contrib.layers.xavier_initializer())

        w_conv3 = tf.get_variable(network_name + 'w_conv3', [3,3,64,128], initializer = tf.contrib.layers.xavier_initializer())
        w_conv4 = tf.get_variable(network_name + 'w_conv4', [3,3,128,128], initializer = tf.contrib.layers.xavier_initializer())

        w_conv5 = tf.get_variable(network_name + 'w_conv5', [3,3,128,256], initializer = tf.contrib.layers.xavier_initializer())
        w_conv6 = tf.get_variable(network_name + 'w_conv6', [3,3,256,256], initializer = tf.contrib.layers.xavier_initializer())
        w_conv7 = tf.get_variable(network_name + 'w_conv7', [3,3,256,256], initializer = tf.contrib.layers.xavier_initializer())

        w_conv8 = tf.get_variable(network_name + 'w_conv8', [3,3,256,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv9 = tf.get_variable(network_name + 'w_conv9', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv10 = tf.get_variable(network_name + 'w_conv10', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())

        w_conv11 = tf.get_variable(network_name + 'w_conv11', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv12 = tf.get_variable(network_name + 'w_conv12', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv13 = tf.get_variable(network_name + 'w_conv13', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())

        w0 = tf.get_variable(network_name + 'w_fc1', [1*1*512, 1024], initializer = tf.contrib.layers.xavier_initializer())
        b0 = tf.get_variable(network_name + 'b_fc1', [1,1024], initializer = tf.contrib.layers.xavier_initializer())
        w1 = tf.get_variable(network_name + 'w_fc2', [1024, 512], initializer = tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(network_name + 'b_fc2', [1,512], initializer = tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable(network_name + 'w_fc3', [512, 10], initializer = tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(network_name + 'b_fc3', [1,10], initializer = tf.contrib.layers.xavier_initializer())

        # network
        con1 = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1 = batch_norm(tf.nn.relu(con1), scale1, beta1, pop_mean1, pop_var1,training=dd)
        con2 = tf.nn.conv2d(h_conv1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = batch_norm(tf.nn.relu(con2), scale2, beta2, pop_mean2, pop_var2,training=dd)
        h_pool2 = max_pool_2x2(h_conv2)
        con3 = tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3 = batch_norm(tf.nn.relu(con3), scale3, beta3, pop_mean3, pop_var3,training=dd)
        con4 = tf.nn.conv2d(h_conv3, w_conv4, strides=[1, 1, 1, 1], padding='SAME')
        h_conv4 = batch_norm(tf.nn.relu(con4), scale4, beta4, pop_mean4, pop_var4,training=dd)
        h_pool4 = max_pool_2x2(h_conv4)
        con5 = tf.nn.conv2d(h_pool4, w_conv5, strides=[1, 1, 1, 1], padding='SAME')
        h_conv5 = batch_norm(tf.nn.relu(con5), scale5, beta5, pop_mean5, pop_var5,training=dd)
        con6 = tf.nn.conv2d(h_conv5, w_conv6, strides=[1, 1, 1, 1], padding='SAME')
        h_conv6 = batch_norm(tf.nn.relu(con6), scale6, beta6, pop_mean6, pop_var6,training=dd)
        con7 = tf.nn.conv2d(h_conv6, w_conv7, strides=[1, 1, 1, 1], padding='SAME')
        h_conv7 = batch_norm(tf.nn.relu(con7), scale7, beta7, pop_mean7, pop_var7,training=dd)
        h_pool7 = max_pool_2x2(h_conv7)
        con8 = tf.nn.conv2d(h_pool7, w_conv8, strides=[1, 1, 1, 1], padding='SAME')
        h_conv8 = batch_norm(tf.nn.relu(con8), scale8, beta8, pop_mean8, pop_var8,training=dd)
        con9 = tf.nn.conv2d(h_conv8, w_conv9, strides=[1, 1, 1, 1], padding='SAME')
        h_conv9 = batch_norm(tf.nn.relu(con9), scale9, beta9, pop_mean9, pop_var9,training=dd)
        con10 = tf.nn.conv2d(h_conv9, w_conv10, strides=[1, 1, 1, 1], padding='SAME')
        h_conv10 = batch_norm(tf.nn.relu(con10), scale10, beta10, pop_mean10, pop_var10,training=dd)
        h_pool10 = max_pool_2x2(h_conv10)
        con11 = tf.nn.conv2d(h_pool10, w_conv11, strides=[1, 1, 1, 1], padding='SAME')
        h_conv11 = batch_norm(tf.nn.relu(con11), scale11, beta11, pop_mean11, pop_var11,training=dd)
        con12 = tf.nn.conv2d(h_conv11, w_conv12, strides=[1, 1, 1, 1], padding='SAME')
        h_conv12 = batch_norm(tf.nn.relu(con12), scale12, beta12, pop_mean12, pop_var12,training=dd)
        con13 = tf.nn.conv2d(h_conv12, w_conv13, strides=[1, 1, 1, 1], padding='SAME')
        h_conv13 = batch_norm(tf.nn.relu(con13), scale13, beta13, pop_mean13, pop_var13,training=dd)
        h_pool13 = max_pool_2x2(h_conv13)
        h_pool_flat = tf.layers.flatten(h_pool13)
        h = tf.nn.relu(tf.matmul(h_pool_flat, w0) + b0)
        h = tf.nn.relu(tf.matmul(h, w1) + b1)
        logits = tf.matmul(h, w2)+b2
        output = tf.nn.softmax(logits)
        probs = tf.reduce_sum(output*y,1)
        cross_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=y))
        regularization = weight_decay*(tf.reduce_sum(tf.square(w0))+tf.reduce_sum(tf.square(b0))
                             +tf.reduce_sum(tf.square(w1))+tf.reduce_sum(tf.square(b1))
                             +tf.reduce_sum(tf.square(w2))+tf.reduce_sum(tf.square(b2))
                             +tf.reduce_sum(tf.square(w_conv1))+tf.reduce_sum(tf.square(w_conv2))
                             +tf.reduce_sum(tf.square(w_conv3))+tf.reduce_sum(tf.square(w_conv4))                                 +tf.reduce_sum(tf.square(w_conv5))+tf.reduce_sum(tf.square(w_conv6))+tf.reduce_sum(tf.square(w_conv7))                                +tf.reduce_sum(tf.square(w_conv8))+tf.reduce_sum(tf.square(w_conv9))+tf.reduce_sum(tf.square(w_conv10))                                +tf.reduce_sum(tf.square(w_conv11))+tf.reduce_sum(tf.square(w_conv12))+tf.reduce_sum(tf.square(w_conv13)))

        loss = cross_ent + regularization
        ent = tf.reduce_sum(-tf.log(output+1e-11)*output,1)

    return x, y, output, probs,ent,loss
  
def vgg16_1(x,network_name,dd,stop_gradient,XX_list,YY_list):
    with tf.variable_scope(network_name,reuse=tf.AUTO_REUSE):
        w_conv1 = tf.get_variable(network_name + 'w_conv1', [3,3,3,64], initializer = tf.contrib.layers.xavier_initializer())
        w_conv2 = tf.get_variable(network_name + 'w_conv2', [3,3,64,64], initializer = tf.contrib.layers.xavier_initializer())
        w_conv3 = tf.get_variable(network_name + 'w_conv3', [3,3,64,128], initializer = tf.contrib.layers.xavier_initializer())
        w_conv4 = tf.get_variable(network_name + 'w_conv4', [3,3,128,128], initializer = tf.contrib.layers.xavier_initializer())
        w_conv5 = tf.get_variable(network_name + 'w_conv5', [3,3,128,256], initializer = tf.contrib.layers.xavier_initializer())
        w_conv6 = tf.get_variable(network_name + 'w_conv6', [3,3,256,256], initializer = tf.contrib.layers.xavier_initializer())
        w_conv7 = tf.get_variable(network_name + 'w_conv7', [3,3,256,256], initializer = tf.contrib.layers.xavier_initializer())
        w_conv8 = tf.get_variable(network_name + 'w_conv8', [3,3,256,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv9 = tf.get_variable(network_name + 'w_conv9', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv10 = tf.get_variable(network_name + 'w_conv10', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv11 = tf.get_variable(network_name + 'w_conv11', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv12 = tf.get_variable(network_name + 'w_conv12', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv13 = tf.get_variable(network_name + 'w_conv13', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w0 = tf.get_variable(network_name + 'w_fc1', [1*1*512, 1024], initializer = tf.contrib.layers.xavier_initializer())
        b0 = tf.get_variable(network_name + 'b_fc1', [1,1024], initializer = tf.contrib.layers.xavier_initializer())
        w1 = tf.get_variable(network_name + 'w_fc2', [1024, 512], initializer = tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(network_name + 'b_fc2', [1,512], initializer = tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable(network_name + 'w_fc3', [512, 10], initializer = tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(network_name + 'b_fc3', [1,10], initializer = tf.contrib.layers.xavier_initializer())
        pop_var1 = tf.get_variable(network_name + "pop_var1", [64], trainable=False,initializer=tf.ones_initializer())
        pop_mean1 = tf.get_variable(network_name + "pop_mean1",[64], trainable=False,initializer=tf.zeros_initializer() )
        pop_var2 = tf.get_variable(network_name + "pop_var2",[64], trainable=False,initializer=tf.ones_initializer())
        pop_mean2 = tf.get_variable(network_name + "pop_mean2",[64], trainable=False,initializer=tf.zeros_initializer() )
        pop_var3 = tf.get_variable(network_name + "pop_var3",[128], trainable=False,initializer=tf.ones_initializer() )
        pop_mean3 = tf.get_variable(network_name + "pop_mean3",[128], trainable=False,initializer=tf.zeros_initializer() )
        pop_var4 = tf.get_variable(network_name + "pop_var4",[128], trainable=False,initializer=tf.ones_initializer())
        pop_mean4 = tf.get_variable(network_name + "pop_mean4",[128], trainable=False,initializer=tf.zeros_initializer())
        pop_var5 = tf.get_variable(network_name + "pop_var5",[256], trainable=False,initializer=tf.ones_initializer() )
        pop_mean5 = tf.get_variable(network_name + "pop_mean5",[256], trainable=False,initializer=tf.zeros_initializer())
        pop_var6 = tf.get_variable(network_name + "pop_var6",[256], trainable=False,initializer=tf.ones_initializer())
        pop_mean6 = tf.get_variable(network_name + "pop_mean6",[256], trainable=False,initializer=tf.zeros_initializer())
        pop_var7 = tf.get_variable(network_name + "pop_var7",[256], trainable=False,initializer=tf.ones_initializer() )
        pop_mean7 = tf.get_variable(network_name + "pop_mean7",[256], trainable=False,initializer=tf.zeros_initializer())
        pop_var8 = tf.get_variable(network_name + "pop_var8",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean8 = tf.get_variable(network_name + "pop_mean8",[512], trainable=False,initializer=tf.zeros_initializer() )
        pop_var9 = tf.get_variable(network_name + "pop_var9",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean9 = tf.get_variable(network_name + "pop_mean9",[512], trainable=False,initializer=tf.zeros_initializer())
        pop_var10 = tf.get_variable(network_name + "pop_var10",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean10 =tf.get_variable( network_name + "pop_mean10",[512], trainable=False,initializer=tf.zeros_initializer())
        pop_var11 = tf.get_variable( network_name + "pop_var11",[512], trainable=False,initializer=tf.ones_initializer())
        pop_mean11 = tf.get_variable(network_name + "pop_mean11",[512], trainable=False,initializer=tf.zeros_initializer() )
        pop_var12 = tf.get_variable( network_name + "pop_var12",[512], trainable=False,initializer=tf.ones_initializer())
        pop_mean12 = tf.get_variable(network_name + "pop_mean12",[512], trainable=False,initializer=tf.zeros_initializer() )
        pop_var13 = tf.get_variable(network_name + "pop_var13",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean13 = tf.get_variable(network_name + "pop_mean13",[512], trainable=False,initializer=tf.zeros_initializer() )
        scale1 = tf.get_variable(network_name + "scale1",[64],initializer=tf.ones_initializer())
        beta1 = tf.get_variable(network_name + "beta1",[64],initializer=tf.zeros_initializer())
        scale2 = tf.get_variable(network_name + "scale2",[64],initializer=tf.ones_initializer())
        beta2 = tf.get_variable(network_name + "beta2",[64],initializer=tf.zeros_initializer())
        scale3 = tf.get_variable(network_name + "scale3",[128],initializer=tf.ones_initializer() )
        beta3 = tf.get_variable(network_name + "beta3",[128],initializer=tf.zeros_initializer() )
        scale4 = tf.get_variable(network_name + "scale4",[128],initializer=tf.ones_initializer() )
        beta4 = tf.get_variable(network_name + "beta4",[128],initializer=tf.zeros_initializer() )
        scale5 = tf.get_variable( network_name + "scale5",[256],initializer=tf.ones_initializer())
        beta5 = tf.get_variable(network_name + "beta5",[256],initializer=tf.zeros_initializer())
        scale6 = tf.get_variable(network_name + "scale6",[256],initializer=tf.ones_initializer() )
        beta6 = tf.get_variable(network_name + "beta6",[256],initializer=tf.zeros_initializer() )
        scale7 = tf.get_variable(network_name + "scale7",[256],initializer=tf.ones_initializer() )
        beta7 = tf.get_variable(network_name + "beta7",[256],initializer=tf.zeros_initializer() )
        scale8 = tf.get_variable(network_name + "scale8",[512],initializer=tf.ones_initializer())
        beta8 = tf.get_variable(network_name + "beta8",[512],initializer=tf.zeros_initializer() )
        scale9 = tf.get_variable(network_name + "scale9",[512],initializer=tf.ones_initializer())
        beta9 = tf.get_variable(network_name + "beta9",[512],initializer=tf.zeros_initializer() )
        scale10 = tf.get_variable(network_name + "scale10",[512],initializer=tf.ones_initializer())
        beta10 = tf.get_variable(network_name + "beta10",[512],initializer=tf.zeros_initializer() )
        scale11 = tf.get_variable(network_name + "scale11",[512],initializer=tf.ones_initializer() )
        beta11 = tf.get_variable(network_name + "beta11",[512],initializer=tf.zeros_initializer())
        scale12 = tf.get_variable(network_name + "scale12",[512],initializer=tf.ones_initializer())
        beta12 = tf.get_variable(network_name + "beta12",[512],initializer=tf.zeros_initializer() )
        scale13 = tf.get_variable( network_name + "scale13",[512],initializer=tf.ones_initializer())
        beta13 = tf.get_variable(network_name + "beta13",[512],initializer=tf.zeros_initializer() )
        scale1s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale1)),lambda:tf.identity(scale1))
        beta1s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta1)),lambda:tf.identity(beta1))
        scale2s = tf.cond(stop_gradient,lambda:tf.stop_gradient(tf.identity(scale2)),lambda:tf.identity(scale2))
        beta2s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta2)),lambda:tf.identity(beta2))
        scale3s = tf.cond(stop_gradient,lambda:tf.stop_gradient(tf.identity(scale3)),lambda:tf.identity(scale3))
        beta3s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta3)),lambda:tf.identity(beta3))
        scale4s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale4)),lambda:tf.identity(scale4))
        beta4s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta4)),lambda:tf.identity(beta4))
        scale5s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale5)),lambda:tf.identity(scale5))
        beta5s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta5)),lambda:tf.identity(beta5))
        scale6s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale6)),lambda:tf.identity(scale6))
        beta6s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta6)),lambda:tf.identity(beta6))
        scale7s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale7)),lambda:tf.identity(scale7))
        beta7s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta7)),lambda:tf.identity(beta7))
        scale8s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale8)),lambda:tf.identity(scale8))
        beta8s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta8)),lambda:tf.identity(beta8))
        scale9s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale9)),lambda:tf.identity(scale9))
        beta9s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta9)),lambda:tf.identity(beta9))
        scale10s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale10)),lambda:tf.identity(scale10))
        beta10s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta10)),lambda:tf.identity(beta10))
        scale11s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale11)),lambda:tf.identity(scale11))
        beta11s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta11)),lambda:tf.identity(beta11))
        scale12s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale12)),lambda:tf.identity(scale12))
        beta12s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta12)),lambda:tf.identity(beta12))
        scale13s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale13)),lambda:tf.identity(scale13))
        beta13s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta13)),lambda:tf.identity(beta13))

        con1 = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1 = batch_norm(tf.nn.relu(con1), scale1s, beta1s, pop_mean1, pop_var1,training=dd)
        con2 = tf.nn.conv2d(h_conv1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = batch_norm(tf.nn.relu(con2), scale2s, beta2s, pop_mean2, pop_var2,training=dd)
        h_pool2 = max_pool_2x2(h_conv2)
        con3 = tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3 = batch_norm(tf.nn.relu(con3), scale3s, beta3s, pop_mean3, pop_var3,training=dd)
        con4 = tf.nn.conv2d(h_conv3, w_conv4, strides=[1, 1, 1, 1], padding='SAME')
        h_conv4 = batch_norm(tf.nn.relu(con4), scale4s, beta4s, pop_mean4, pop_var4,training=dd)
        h_pool4 = max_pool_2x2(h_conv4)
        con5 = tf.nn.conv2d(h_pool4, w_conv5, strides=[1, 1, 1, 1], padding='SAME')
        h_conv5 = batch_norm(tf.nn.relu(con5), scale5s, beta5s, pop_mean5, pop_var5,training=dd)
        con6 = tf.nn.conv2d(h_conv5, w_conv6, strides=[1, 1, 1, 1], padding='SAME')
        h_conv6 = batch_norm(tf.nn.relu(con6), scale6s, beta6s, pop_mean6, pop_var6,training=dd)
        con7 = tf.nn.conv2d(h_conv6, w_conv7, strides=[1, 1, 1, 1], padding='SAME')
        h_conv7 = batch_norm(tf.nn.relu(con7), scale7s, beta7s, pop_mean7, pop_var7,training=dd)
        h_pool7 = max_pool_2x2(h_conv7)
        con8 = tf.nn.conv2d(h_pool7, w_conv8, strides=[1, 1, 1, 1], padding='SAME')
        h_conv8 = batch_norm(tf.nn.relu(con8), scale8s, beta8s, pop_mean8, pop_var8,training=dd)
        con9 = tf.nn.conv2d(h_conv8, w_conv9, strides=[1, 1, 1, 1], padding='SAME')
        h_conv9 = batch_norm(tf.nn.relu(con9), scale9s, beta9s, pop_mean9, pop_var9,training=dd)
        con10 = tf.nn.conv2d(h_conv9, w_conv10, strides=[1, 1, 1, 1], padding='SAME')
        h_conv10 = batch_norm(tf.nn.relu(con10), scale10s, beta10s, pop_mean10, pop_var10,training=dd)
        h_pool10 = max_pool_2x2(h_conv10)
        con11 = tf.nn.conv2d(h_pool10, w_conv11, strides=[1, 1, 1, 1], padding='SAME')
        h_conv11 = batch_norm(tf.nn.relu(con11), scale11s, beta11s, pop_mean11, pop_var11,training=dd)
        con12 = tf.nn.conv2d(h_conv11, w_conv12, strides=[1, 1, 1, 1], padding='SAME')
        h_conv12 = batch_norm(tf.nn.relu(con12), scale12s, beta12s, pop_mean12, pop_var12,training=dd)
        con13 = tf.nn.conv2d(h_conv12, w_conv13, strides=[1, 1, 1, 1], padding='SAME')
        h_conv13 = batch_norm(tf.nn.relu(con13), scale13s, beta13s, pop_mean13, pop_var13,training=dd)
        h_pool13 = max_pool_2x2(h_conv13)
        h_pool_flat = tf.layers.flatten(h_pool13)
        h = tf.nn.relu(tf.matmul(h_pool_flat, w0) + b0)
        h = tf.nn.relu(tf.matmul(h, w1) + b1)
        logits = tf.matmul(h, w2)+b2
        output = tf.nn.softmax(logits)
    return x, output

def vgg16_cifar100(network_name,dd,XX_list,YY_list,weight_decay): 
    with tf.variable_scope(network_name,reuse=tf.AUTO_REUSE):
        x = XX_list[network_name]
        y = YY_list[network_name]
        pop_var1 = tf.get_variable(network_name + "pop_var1", [64], trainable=False,initializer=tf.ones_initializer())
        pop_mean1 = tf.get_variable(network_name + "pop_mean1",[64], trainable=False,initializer=tf.zeros_initializer() )
        pop_var2 = tf.get_variable(network_name + "pop_var2",[64], trainable=False,initializer=tf.ones_initializer())
        pop_mean2 = tf.get_variable(network_name + "pop_mean2",[64], trainable=False,initializer=tf.zeros_initializer() )
        pop_var3 = tf.get_variable(network_name + "pop_var3",[128], trainable=False,initializer=tf.ones_initializer() )
        pop_mean3 = tf.get_variable(network_name + "pop_mean3",[128], trainable=False,initializer=tf.zeros_initializer() )
        pop_var4 = tf.get_variable(network_name + "pop_var4",[128], trainable=False,initializer=tf.ones_initializer())
        pop_mean4 = tf.get_variable(network_name + "pop_mean4",[128], trainable=False,initializer=tf.zeros_initializer())
        pop_var5 = tf.get_variable(network_name + "pop_var5",[256], trainable=False,initializer=tf.ones_initializer() )
        pop_mean5 = tf.get_variable(network_name + "pop_mean5",[256], trainable=False,initializer=tf.zeros_initializer())
        pop_var6 = tf.get_variable(network_name + "pop_var6",[256], trainable=False,initializer=tf.ones_initializer())
        pop_mean6 = tf.get_variable(network_name + "pop_mean6",[256], trainable=False,initializer=tf.zeros_initializer())
        pop_var7 = tf.get_variable(network_name + "pop_var7",[256], trainable=False,initializer=tf.ones_initializer() )
        pop_mean7 = tf.get_variable(network_name + "pop_mean7",[256], trainable=False,initializer=tf.zeros_initializer())
        pop_var8 = tf.get_variable(network_name + "pop_var8",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean8 = tf.get_variable(network_name + "pop_mean8",[512], trainable=False,initializer=tf.zeros_initializer() )
        pop_var9 = tf.get_variable(network_name + "pop_var9",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean9 = tf.get_variable(network_name + "pop_mean9",[512], trainable=False,initializer=tf.zeros_initializer())
        pop_var10 = tf.get_variable(network_name + "pop_var10",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean10 =tf.get_variable( network_name + "pop_mean10",[512], trainable=False,initializer=tf.zeros_initializer())
        pop_var11 = tf.get_variable( network_name + "pop_var11",[512], trainable=False,initializer=tf.ones_initializer())
        pop_mean11 = tf.get_variable(network_name + "pop_mean11",[512], trainable=False,initializer=tf.zeros_initializer() )
        pop_var12 = tf.get_variable( network_name + "pop_var12",[512], trainable=False,initializer=tf.ones_initializer())
        pop_mean12 = tf.get_variable(network_name + "pop_mean12",[512], trainable=False,initializer=tf.zeros_initializer() )
        pop_var13 = tf.get_variable(network_name + "pop_var13",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean13 = tf.get_variable(network_name + "pop_mean13",[512], trainable=False,initializer=tf.zeros_initializer() )

        scale1 = tf.get_variable(network_name + "scale1",[64],initializer=tf.ones_initializer())
        beta1 = tf.get_variable(network_name + "beta1",[64],initializer=tf.zeros_initializer())
        scale2 = tf.get_variable(network_name + "scale2",[64],initializer=tf.ones_initializer())
        beta2 = tf.get_variable(network_name + "beta2",[64],initializer=tf.zeros_initializer())
        scale3 = tf.get_variable(network_name + "scale3",[128],initializer=tf.ones_initializer() )
        beta3 = tf.get_variable(network_name + "beta3",[128],initializer=tf.zeros_initializer() )
        scale4 = tf.get_variable(network_name + "scale4",[128],initializer=tf.ones_initializer() )
        beta4 = tf.get_variable(network_name + "beta4",[128],initializer=tf.zeros_initializer() )
        scale5 = tf.get_variable( network_name + "scale5",[256],initializer=tf.ones_initializer())
        beta5 = tf.get_variable(network_name + "beta5",[256],initializer=tf.zeros_initializer())
        scale6 = tf.get_variable(network_name + "scale6",[256],initializer=tf.ones_initializer() )
        beta6 = tf.get_variable(network_name + "beta6",[256],initializer=tf.zeros_initializer() )
        scale7 = tf.get_variable(network_name + "scale7",[256],initializer=tf.ones_initializer() )
        beta7 = tf.get_variable(network_name + "beta7",[256],initializer=tf.zeros_initializer() )
        scale8 = tf.get_variable(network_name + "scale8",[512],initializer=tf.ones_initializer())
        beta8 = tf.get_variable(network_name + "beta8",[512],initializer=tf.zeros_initializer() )
        scale9 = tf.get_variable(network_name + "scale9",[512],initializer=tf.ones_initializer())
        beta9 = tf.get_variable(network_name + "beta9",[512],initializer=tf.zeros_initializer() )
        scale10 = tf.get_variable(network_name + "scale10",[512],initializer=tf.ones_initializer())
        beta10 = tf.get_variable(network_name + "beta10",[512],initializer=tf.zeros_initializer() )
        scale11 = tf.get_variable(network_name + "scale11",[512],initializer=tf.ones_initializer() )
        beta11 = tf.get_variable(network_name + "beta11",[512],initializer=tf.zeros_initializer())
        scale12 = tf.get_variable(network_name + "scale12",[512],initializer=tf.ones_initializer())
        beta12 = tf.get_variable(network_name + "beta12",[512],initializer=tf.zeros_initializer() )
        scale13 = tf.get_variable( network_name + "scale13",[512],initializer=tf.ones_initializer())
        beta13 = tf.get_variable(network_name + "beta13",[512],initializer=tf.zeros_initializer() )

        w_conv1 = tf.get_variable(network_name + 'w_conv1', [3,3,3,64], initializer = tf.contrib.layers.xavier_initializer())
        w_conv2 = tf.get_variable(network_name + 'w_conv2', [3,3,64,64], initializer = tf.contrib.layers.xavier_initializer())

        w_conv3 = tf.get_variable(network_name + 'w_conv3', [3,3,64,128], initializer = tf.contrib.layers.xavier_initializer())
        w_conv4 = tf.get_variable(network_name + 'w_conv4', [3,3,128,128], initializer = tf.contrib.layers.xavier_initializer())

        w_conv5 = tf.get_variable(network_name + 'w_conv5', [3,3,128,256], initializer = tf.contrib.layers.xavier_initializer())
        w_conv6 = tf.get_variable(network_name + 'w_conv6', [3,3,256,256], initializer = tf.contrib.layers.xavier_initializer())
        w_conv7 = tf.get_variable(network_name + 'w_conv7', [3,3,256,256], initializer = tf.contrib.layers.xavier_initializer())

        w_conv8 = tf.get_variable(network_name + 'w_conv8', [3,3,256,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv9 = tf.get_variable(network_name + 'w_conv9', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv10 = tf.get_variable(network_name + 'w_conv10', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())

        w_conv11 = tf.get_variable(network_name + 'w_conv11', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv12 = tf.get_variable(network_name + 'w_conv12', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv13 = tf.get_variable(network_name + 'w_conv13', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())

        w0 = tf.get_variable(network_name + 'w_fc1', [1*1*512, 1024], initializer = tf.contrib.layers.xavier_initializer())
        b0 = tf.get_variable(network_name + 'b_fc1', [1,1024], initializer = tf.contrib.layers.xavier_initializer())
        w1 = tf.get_variable(network_name + 'w_fc2', [1024, 512], initializer = tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(network_name + 'b_fc2', [1,512], initializer = tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable(network_name + 'w_fc3', [512, 100], initializer = tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(network_name + 'b_fc3', [1,100], initializer = tf.contrib.layers.xavier_initializer())

        # network
        con1 = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1 = batch_norm(tf.nn.relu(con1), scale1, beta1, pop_mean1, pop_var1,training=dd)
        con2 = tf.nn.conv2d(h_conv1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = batch_norm(tf.nn.relu(con2), scale2, beta2, pop_mean2, pop_var2,training=dd)
        h_pool2 = max_pool_2x2(h_conv2)
        con3 = tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3 = batch_norm(tf.nn.relu(con3), scale3, beta3, pop_mean3, pop_var3,training=dd)
        con4 = tf.nn.conv2d(h_conv3, w_conv4, strides=[1, 1, 1, 1], padding='SAME')
        h_conv4 = batch_norm(tf.nn.relu(con4), scale4, beta4, pop_mean4, pop_var4,training=dd)
        h_pool4 = max_pool_2x2(h_conv4)
        con5 = tf.nn.conv2d(h_pool4, w_conv5, strides=[1, 1, 1, 1], padding='SAME')
        h_conv5 = batch_norm(tf.nn.relu(con5), scale5, beta5, pop_mean5, pop_var5,training=dd)
        con6 = tf.nn.conv2d(h_conv5, w_conv6, strides=[1, 1, 1, 1], padding='SAME')
        h_conv6 = batch_norm(tf.nn.relu(con6), scale6, beta6, pop_mean6, pop_var6,training=dd)
        con7 = tf.nn.conv2d(h_conv6, w_conv7, strides=[1, 1, 1, 1], padding='SAME')
        h_conv7 = batch_norm(tf.nn.relu(con7), scale7, beta7, pop_mean7, pop_var7,training=dd)
        h_pool7 = max_pool_2x2(h_conv7)
        con8 = tf.nn.conv2d(h_pool7, w_conv8, strides=[1, 1, 1, 1], padding='SAME')
        h_conv8 = batch_norm(tf.nn.relu(con8), scale8, beta8, pop_mean8, pop_var8,training=dd)
        con9 = tf.nn.conv2d(h_conv8, w_conv9, strides=[1, 1, 1, 1], padding='SAME')
        h_conv9 = batch_norm(tf.nn.relu(con9), scale9, beta9, pop_mean9, pop_var9,training=dd)
        con10 = tf.nn.conv2d(h_conv9, w_conv10, strides=[1, 1, 1, 1], padding='SAME')
        h_conv10 = batch_norm(tf.nn.relu(con10), scale10, beta10, pop_mean10, pop_var10,training=dd)
        h_pool10 = max_pool_2x2(h_conv10)
        con11 = tf.nn.conv2d(h_pool10, w_conv11, strides=[1, 1, 1, 1], padding='SAME')
        h_conv11 = batch_norm(tf.nn.relu(con11), scale11, beta11, pop_mean11, pop_var11,training=dd)
        con12 = tf.nn.conv2d(h_conv11, w_conv12, strides=[1, 1, 1, 1], padding='SAME')
        h_conv12 = batch_norm(tf.nn.relu(con12), scale12, beta12, pop_mean12, pop_var12,training=dd)
        con13 = tf.nn.conv2d(h_conv12, w_conv13, strides=[1, 1, 1, 1], padding='SAME')
        h_conv13 = batch_norm(tf.nn.relu(con13), scale13, beta13, pop_mean13, pop_var13,training=dd)
        h_pool13 = max_pool_2x2(h_conv13)
        h_pool_flat = tf.layers.flatten(h_pool13)
        h = tf.nn.relu(tf.matmul(h_pool_flat, w0) + b0)
        h = tf.nn.relu(tf.matmul(h, w1) + b1)
        logits = tf.matmul(h, w2)+b2
        output = tf.nn.softmax(logits)
        probs = tf.reduce_sum(output*y,1)
        cross_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=y))
        regularization = weight_decay*(tf.reduce_sum(tf.square(w0))+tf.reduce_sum(tf.square(b0))
                             +tf.reduce_sum(tf.square(w1))+tf.reduce_sum(tf.square(b1))
                             +tf.reduce_sum(tf.square(w2))+tf.reduce_sum(tf.square(b2))
                             +tf.reduce_sum(tf.square(w_conv1))+tf.reduce_sum(tf.square(w_conv2))
                             +tf.reduce_sum(tf.square(w_conv3))+tf.reduce_sum(tf.square(w_conv4))                                 +tf.reduce_sum(tf.square(w_conv5))+tf.reduce_sum(tf.square(w_conv6))+tf.reduce_sum(tf.square(w_conv7))                                +tf.reduce_sum(tf.square(w_conv8))+tf.reduce_sum(tf.square(w_conv9))+tf.reduce_sum(tf.square(w_conv10))                                +tf.reduce_sum(tf.square(w_conv11))+tf.reduce_sum(tf.square(w_conv12))+tf.reduce_sum(tf.square(w_conv13)))

        loss = cross_ent + regularization
        ent = tf.reduce_sum(-tf.log(output+1e-11)*output,1)

    return x, y, output, probs,ent,loss
  
def vgg16_cifar100_1(x,network_name,dd,stop_gradient,XX_list,YY_list):
    with tf.variable_scope(network_name,reuse=tf.AUTO_REUSE):
        w_conv1 = tf.get_variable(network_name + 'w_conv1', [3,3,3,64], initializer = tf.contrib.layers.xavier_initializer())
        w_conv2 = tf.get_variable(network_name + 'w_conv2', [3,3,64,64], initializer = tf.contrib.layers.xavier_initializer())
        w_conv3 = tf.get_variable(network_name + 'w_conv3', [3,3,64,128], initializer = tf.contrib.layers.xavier_initializer())
        w_conv4 = tf.get_variable(network_name + 'w_conv4', [3,3,128,128], initializer = tf.contrib.layers.xavier_initializer())
        w_conv5 = tf.get_variable(network_name + 'w_conv5', [3,3,128,256], initializer = tf.contrib.layers.xavier_initializer())
        w_conv6 = tf.get_variable(network_name + 'w_conv6', [3,3,256,256], initializer = tf.contrib.layers.xavier_initializer())
        w_conv7 = tf.get_variable(network_name + 'w_conv7', [3,3,256,256], initializer = tf.contrib.layers.xavier_initializer())
        w_conv8 = tf.get_variable(network_name + 'w_conv8', [3,3,256,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv9 = tf.get_variable(network_name + 'w_conv9', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv10 = tf.get_variable(network_name + 'w_conv10', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv11 = tf.get_variable(network_name + 'w_conv11', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv12 = tf.get_variable(network_name + 'w_conv12', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w_conv13 = tf.get_variable(network_name + 'w_conv13', [3,3,512,512], initializer = tf.contrib.layers.xavier_initializer())
        w0 = tf.get_variable(network_name + 'w_fc1', [1*1*512, 1024], initializer = tf.contrib.layers.xavier_initializer())
        b0 = tf.get_variable(network_name + 'b_fc1', [1,1024], initializer = tf.contrib.layers.xavier_initializer())
        w1 = tf.get_variable(network_name + 'w_fc2', [1024, 512], initializer = tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(network_name + 'b_fc2', [1,512], initializer = tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable(network_name + 'w_fc3', [512, 100], initializer = tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(network_name + 'b_fc3', [1,100], initializer = tf.contrib.layers.xavier_initializer())
        pop_var1 = tf.get_variable(network_name + "pop_var1", [64], trainable=False,initializer=tf.ones_initializer())
        pop_mean1 = tf.get_variable(network_name + "pop_mean1",[64], trainable=False,initializer=tf.zeros_initializer() )
        pop_var2 = tf.get_variable(network_name + "pop_var2",[64], trainable=False,initializer=tf.ones_initializer())
        pop_mean2 = tf.get_variable(network_name + "pop_mean2",[64], trainable=False,initializer=tf.zeros_initializer() )
        pop_var3 = tf.get_variable(network_name + "pop_var3",[128], trainable=False,initializer=tf.ones_initializer() )
        pop_mean3 = tf.get_variable(network_name + "pop_mean3",[128], trainable=False,initializer=tf.zeros_initializer() )
        pop_var4 = tf.get_variable(network_name + "pop_var4",[128], trainable=False,initializer=tf.ones_initializer())
        pop_mean4 = tf.get_variable(network_name + "pop_mean4",[128], trainable=False,initializer=tf.zeros_initializer())
        pop_var5 = tf.get_variable(network_name + "pop_var5",[256], trainable=False,initializer=tf.ones_initializer() )
        pop_mean5 = tf.get_variable(network_name + "pop_mean5",[256], trainable=False,initializer=tf.zeros_initializer())
        pop_var6 = tf.get_variable(network_name + "pop_var6",[256], trainable=False,initializer=tf.ones_initializer())
        pop_mean6 = tf.get_variable(network_name + "pop_mean6",[256], trainable=False,initializer=tf.zeros_initializer())
        pop_var7 = tf.get_variable(network_name + "pop_var7",[256], trainable=False,initializer=tf.ones_initializer() )
        pop_mean7 = tf.get_variable(network_name + "pop_mean7",[256], trainable=False,initializer=tf.zeros_initializer())
        pop_var8 = tf.get_variable(network_name + "pop_var8",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean8 = tf.get_variable(network_name + "pop_mean8",[512], trainable=False,initializer=tf.zeros_initializer() )
        pop_var9 = tf.get_variable(network_name + "pop_var9",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean9 = tf.get_variable(network_name + "pop_mean9",[512], trainable=False,initializer=tf.zeros_initializer())
        pop_var10 = tf.get_variable(network_name + "pop_var10",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean10 =tf.get_variable( network_name + "pop_mean10",[512], trainable=False,initializer=tf.zeros_initializer())
        pop_var11 = tf.get_variable( network_name + "pop_var11",[512], trainable=False,initializer=tf.ones_initializer())
        pop_mean11 = tf.get_variable(network_name + "pop_mean11",[512], trainable=False,initializer=tf.zeros_initializer() )
        pop_var12 = tf.get_variable( network_name + "pop_var12",[512], trainable=False,initializer=tf.ones_initializer())
        pop_mean12 = tf.get_variable(network_name + "pop_mean12",[512], trainable=False,initializer=tf.zeros_initializer() )
        pop_var13 = tf.get_variable(network_name + "pop_var13",[512], trainable=False,initializer=tf.ones_initializer() )
        pop_mean13 = tf.get_variable(network_name + "pop_mean13",[512], trainable=False,initializer=tf.zeros_initializer() )
        scale1 = tf.get_variable(network_name + "scale1",[64],initializer=tf.ones_initializer())
        beta1 = tf.get_variable(network_name + "beta1",[64],initializer=tf.zeros_initializer())
        scale2 = tf.get_variable(network_name + "scale2",[64],initializer=tf.ones_initializer())
        beta2 = tf.get_variable(network_name + "beta2",[64],initializer=tf.zeros_initializer())
        scale3 = tf.get_variable(network_name + "scale3",[128],initializer=tf.ones_initializer() )
        beta3 = tf.get_variable(network_name + "beta3",[128],initializer=tf.zeros_initializer() )
        scale4 = tf.get_variable(network_name + "scale4",[128],initializer=tf.ones_initializer() )
        beta4 = tf.get_variable(network_name + "beta4",[128],initializer=tf.zeros_initializer() )
        scale5 = tf.get_variable( network_name + "scale5",[256],initializer=tf.ones_initializer())
        beta5 = tf.get_variable(network_name + "beta5",[256],initializer=tf.zeros_initializer())
        scale6 = tf.get_variable(network_name + "scale6",[256],initializer=tf.ones_initializer() )
        beta6 = tf.get_variable(network_name + "beta6",[256],initializer=tf.zeros_initializer() )
        scale7 = tf.get_variable(network_name + "scale7",[256],initializer=tf.ones_initializer() )
        beta7 = tf.get_variable(network_name + "beta7",[256],initializer=tf.zeros_initializer() )
        scale8 = tf.get_variable(network_name + "scale8",[512],initializer=tf.ones_initializer())
        beta8 = tf.get_variable(network_name + "beta8",[512],initializer=tf.zeros_initializer() )
        scale9 = tf.get_variable(network_name + "scale9",[512],initializer=tf.ones_initializer())
        beta9 = tf.get_variable(network_name + "beta9",[512],initializer=tf.zeros_initializer() )
        scale10 = tf.get_variable(network_name + "scale10",[512],initializer=tf.ones_initializer())
        beta10 = tf.get_variable(network_name + "beta10",[512],initializer=tf.zeros_initializer() )
        scale11 = tf.get_variable(network_name + "scale11",[512],initializer=tf.ones_initializer() )
        beta11 = tf.get_variable(network_name + "beta11",[512],initializer=tf.zeros_initializer())
        scale12 = tf.get_variable(network_name + "scale12",[512],initializer=tf.ones_initializer())
        beta12 = tf.get_variable(network_name + "beta12",[512],initializer=tf.zeros_initializer() )
        scale13 = tf.get_variable( network_name + "scale13",[512],initializer=tf.ones_initializer())
        beta13 = tf.get_variable(network_name + "beta13",[512],initializer=tf.zeros_initializer() )
        scale1s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale1)),lambda:tf.identity(scale1))
        beta1s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta1)),lambda:tf.identity(beta1))
        scale2s = tf.cond(stop_gradient,lambda:tf.stop_gradient(tf.identity(scale2)),lambda:tf.identity(scale2))
        beta2s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta2)),lambda:tf.identity(beta2))
        scale3s = tf.cond(stop_gradient,lambda:tf.stop_gradient(tf.identity(scale3)),lambda:tf.identity(scale3))
        beta3s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta3)),lambda:tf.identity(beta3))
        scale4s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale4)),lambda:tf.identity(scale4))
        beta4s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta4)),lambda:tf.identity(beta4))
        scale5s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale5)),lambda:tf.identity(scale5))
        beta5s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta5)),lambda:tf.identity(beta5))
        scale6s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale6)),lambda:tf.identity(scale6))
        beta6s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta6)),lambda:tf.identity(beta6))
        scale7s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale7)),lambda:tf.identity(scale7))
        beta7s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta7)),lambda:tf.identity(beta7))
        scale8s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale8)),lambda:tf.identity(scale8))
        beta8s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta8)),lambda:tf.identity(beta8))
        scale9s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale9)),lambda:tf.identity(scale9))
        beta9s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta9)),lambda:tf.identity(beta9))
        scale10s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale10)),lambda:tf.identity(scale10))
        beta10s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta10)),lambda:tf.identity(beta10))
        scale11s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale11)),lambda:tf.identity(scale11))
        beta11s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta11)),lambda:tf.identity(beta11))
        scale12s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale12)),lambda:tf.identity(scale12))
        beta12s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta12)),lambda:tf.identity(beta12))
        scale13s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale13)),lambda:tf.identity(scale13))
        beta13s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta13)),lambda:tf.identity(beta13))

        con1 = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1 = batch_norm(tf.nn.relu(con1), scale1s, beta1s, pop_mean1, pop_var1,training=dd)
        con2 = tf.nn.conv2d(h_conv1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = batch_norm(tf.nn.relu(con2), scale2s, beta2s, pop_mean2, pop_var2,training=dd)
        h_pool2 = max_pool_2x2(h_conv2)
        con3 = tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')
        h_conv3 = batch_norm(tf.nn.relu(con3), scale3s, beta3s, pop_mean3, pop_var3,training=dd)
        con4 = tf.nn.conv2d(h_conv3, w_conv4, strides=[1, 1, 1, 1], padding='SAME')
        h_conv4 = batch_norm(tf.nn.relu(con4), scale4s, beta4s, pop_mean4, pop_var4,training=dd)
        h_pool4 = max_pool_2x2(h_conv4)
        con5 = tf.nn.conv2d(h_pool4, w_conv5, strides=[1, 1, 1, 1], padding='SAME')
        h_conv5 = batch_norm(tf.nn.relu(con5), scale5s, beta5s, pop_mean5, pop_var5,training=dd)
        con6 = tf.nn.conv2d(h_conv5, w_conv6, strides=[1, 1, 1, 1], padding='SAME')
        h_conv6 = batch_norm(tf.nn.relu(con6), scale6s, beta6s, pop_mean6, pop_var6,training=dd)
        con7 = tf.nn.conv2d(h_conv6, w_conv7, strides=[1, 1, 1, 1], padding='SAME')
        h_conv7 = batch_norm(tf.nn.relu(con7), scale7s, beta7s, pop_mean7, pop_var7,training=dd)
        h_pool7 = max_pool_2x2(h_conv7)
        con8 = tf.nn.conv2d(h_pool7, w_conv8, strides=[1, 1, 1, 1], padding='SAME')
        h_conv8 = batch_norm(tf.nn.relu(con8), scale8s, beta8s, pop_mean8, pop_var8,training=dd)
        con9 = tf.nn.conv2d(h_conv8, w_conv9, strides=[1, 1, 1, 1], padding='SAME')
        h_conv9 = batch_norm(tf.nn.relu(con9), scale9s, beta9s, pop_mean9, pop_var9,training=dd)
        con10 = tf.nn.conv2d(h_conv9, w_conv10, strides=[1, 1, 1, 1], padding='SAME')
        h_conv10 = batch_norm(tf.nn.relu(con10), scale10s, beta10s, pop_mean10, pop_var10,training=dd)
        h_pool10 = max_pool_2x2(h_conv10)
        con11 = tf.nn.conv2d(h_pool10, w_conv11, strides=[1, 1, 1, 1], padding='SAME')
        h_conv11 = batch_norm(tf.nn.relu(con11), scale11s, beta11s, pop_mean11, pop_var11,training=dd)
        con12 = tf.nn.conv2d(h_conv11, w_conv12, strides=[1, 1, 1, 1], padding='SAME')
        h_conv12 = batch_norm(tf.nn.relu(con12), scale12s, beta12s, pop_mean12, pop_var12,training=dd)
        con13 = tf.nn.conv2d(h_conv12, w_conv13, strides=[1, 1, 1, 1], padding='SAME')
        h_conv13 = batch_norm(tf.nn.relu(con13), scale13s, beta13s, pop_mean13, pop_var13,training=dd)
        h_pool13 = max_pool_2x2(h_conv13)
        h_pool_flat = tf.layers.flatten(h_pool13)
        h = tf.nn.relu(tf.matmul(h_pool_flat, w0) + b0)
        h = tf.nn.relu(tf.matmul(h, w1) + b1)
        logits = tf.matmul(h, w2)+b2
        output = tf.nn.softmax(logits)
    return x, output

def vgg4(network_name,dd,XX_list,YY_list,weight_decay): 
    with tf.variable_scope(network_name,reuse=tf.AUTO_REUSE):
        x = XX_list[network_name]
        y = YY_list[network_name]
        pop_var1 = tf.get_variable(network_name + "pop_var1", [32], trainable=False,initializer=tf.ones_initializer())
        pop_mean1 = tf.get_variable(network_name + "pop_mean1",[32], trainable=False,initializer=tf.zeros_initializer() )
        pop_var2 = tf.get_variable(network_name + "pop_var2",[64], trainable=False,initializer=tf.ones_initializer())
        pop_mean2 = tf.get_variable(network_name + "pop_mean2",[64], trainable=False,initializer=tf.zeros_initializer() )
        scale1 = tf.get_variable(network_name + "scale1",[32],initializer=tf.ones_initializer())
        beta1 = tf.get_variable(network_name + "beta1",[32],initializer=tf.zeros_initializer())
        scale2 = tf.get_variable(network_name + "scale2",[64],initializer=tf.ones_initializer())
        beta2 = tf.get_variable(network_name + "beta2",[64],initializer=tf.zeros_initializer())

        w_conv1 = tf.get_variable(network_name + 'w_conv1', [5,5, 1,32], initializer = tf.contrib.layers.xavier_initializer())
        w_conv2 = tf.get_variable(network_name + 'w_conv2', [5,5,32,64], initializer = tf.contrib.layers.xavier_initializer())
        w0 = tf.get_variable(network_name + 'w_fc1', [7*7*64, 512], initializer = tf.contrib.layers.xavier_initializer())
        b0 = tf.get_variable(network_name + 'b_fc1', [1,512], initializer = tf.contrib.layers.xavier_initializer())
        w1 = tf.get_variable(network_name + 'w_fc2', [512, 10], initializer = tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(network_name + 'b_fc2', [1,10], initializer = tf.contrib.layers.xavier_initializer())
        con1 = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1 = batch_norm(tf.nn.relu(con1), scale1, beta1, pop_mean1, pop_var1,training=dd)
        h_pool1 = max_pool_2x2(h_conv1)
        con2 = tf.nn.conv2d(h_pool1 , w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = batch_norm(tf.nn.relu(con2), scale2, beta2, pop_mean2, pop_var2,training=dd)
        h_pool2 = max_pool_2x2(h_conv2)            
        h_pool_flat = tf.layers.flatten(h_pool2)
        h = tf.nn.relu(tf.matmul(h_pool_flat, w0) + b0)
        logits = tf.matmul(h, w1)+b1
        output = tf.nn.softmax(logits)
        probs = tf.reduce_sum(output*y,1)
        p=tf.reduce_mean(probs,0)
        cross_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=y))
        regularization = 1e-6*(tf.reduce_sum(tf.square(w0))+tf.reduce_sum(tf.square(b0))
                             +tf.reduce_sum(tf.square(w1))+tf.reduce_sum(tf.square(b1))
                             +tf.reduce_sum(tf.square(w_conv1))+tf.reduce_sum(tf.square(w_conv2)))                                
        loss = cross_ent + regularization
        ent = tf.reduce_sum(-tf.log(output+1e-11)*output,1)
    return x, y, output, probs,ent,loss
  
def vgg4_1(x,network_name,dd,stop_gradient,XX_list,YY_list):

    with tf.variable_scope(network_name,reuse=tf.AUTO_REUSE):
        pop_var1 = tf.get_variable(network_name + "pop_var1", [32], trainable=False,initializer=tf.ones_initializer())
        pop_mean1 = tf.get_variable(network_name + "pop_mean1",[32], trainable=False,initializer=tf.zeros_initializer() )
        pop_var2 = tf.get_variable(network_name + "pop_var2",[64], trainable=False,initializer=tf.ones_initializer())
        pop_mean2 = tf.get_variable(network_name + "pop_mean2",[64], trainable=False,initializer=tf.zeros_initializer() )
        scale1 = tf.get_variable(network_name + "scale1",[32],initializer=tf.ones_initializer())
        beta1 = tf.get_variable(network_name + "beta1",[32],initializer=tf.zeros_initializer())
        scale2 = tf.get_variable(network_name + "scale2",[64],initializer=tf.ones_initializer())
        beta2 = tf.get_variable(network_name + "beta2",[64],initializer=tf.zeros_initializer())            
        w_conv1 = tf.get_variable(network_name + 'w_conv1', [5,5, 1,32], initializer = tf.contrib.layers.xavier_initializer())
        w_conv2 = tf.get_variable(network_name + 'w_conv2', [5,5,32,64], initializer = tf.contrib.layers.xavier_initializer())
        w0 = tf.get_variable(network_name + 'w_fc1', [7*7*64, 512], initializer = tf.contrib.layers.xavier_initializer())
        b0 = tf.get_variable(network_name + 'b_fc1', [1,512], initializer = tf.contrib.layers.xavier_initializer())
        w1 = tf.get_variable(network_name + 'w_fc2', [512, 10], initializer = tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(network_name + 'b_fc2', [1,10], initializer = tf.contrib.layers.xavier_initializer())

        scale1s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(scale1)),lambda:tf.identity(scale1))
        beta1s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta1)),lambda:tf.identity(beta1))
        scale2s = tf.cond(stop_gradient,lambda:tf.stop_gradient(tf.identity(scale2)),lambda:tf.identity(scale2))
        beta2s = tf.cond(stop_gradient, lambda:tf.stop_gradient(tf.identity(beta2)),lambda:tf.identity(beta2))
        con1 = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1 = batch_norm(tf.nn.relu(con1), scale1s, beta1s, pop_mean1, pop_var1,training=dd)
        h_pool1 = max_pool_2x2(h_conv1)
        con2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = batch_norm(tf.nn.relu(con2), scale2s, beta2s, pop_mean2, pop_var2,training=dd)
        h_pool2 = max_pool_2x2(h_conv2)         
        h_pool_flat = tf.layers.flatten(h_pool2)
        h = tf.nn.relu(tf.matmul(h_pool_flat, w0) + b0)
        logits = tf.matmul(h, w1)+b1
        output = tf.nn.softmax(logits)
    return x, output

