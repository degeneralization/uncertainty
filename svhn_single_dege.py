import tensorflow as tf
import numpy as np
import os
import scipy.io
from scipy import io
import random
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import cv2
from utilities import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# LSUN data
LSUN = np.load("./LSUN.npy")
lSUN = LSUN/255.

# TIM data
TIM = np.load("./TIM.npy") 

# cifar10 data
(x_train1, y_train1), (x_test1, y_test1) = tf.keras.datasets.cifar10.load_data()
cifar10 = x_test1/255.

# svhn data

def OneHot(label,n_classes):
    label=np.array(label).reshape(-1)
    label=np.eye(classes)[label]
    return label
data1=io.loadmat('train_32x32.mat')
data2=io.loadmat('test_32x32.mat')

train_data=data1['X']
train_labels=data1['y']
test_data=data2['X']
test_labels=data2['y']

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data = np.transpose(train_data, (3, 0, 1, 2))
test_data = np.transpose(test_data,(3, 0, 1, 2))
train_data = train_data/255.
test_data = test_data/255.

train_labels[train_labels==10]=0
test_labels[test_labels==10]=0
classes=10
train_labels = train_labels[:,0]
test_labels = test_labels[:,0]
train_labels=OneHot(train_labels,classes)
test_labels=OneHot(test_labels,classes)

x_train = train_data
y_train = train_labels
x_test = test_data[:10000]
y_test = test_labels[:10000]

batch_size = 128
tf.reset_default_graph()
networks = ['network1']

XX_list={}
YY_list={}
for mm in range(len(networks)):
  with tf.name_scope(networks[mm]):
    XX_list[networks[mm]] = tf.placeholder(tf.float32, shape = [None, 32, 32, 3],name = networks[mm] +'x')
    YY_list[networks[mm]] = tf.placeholder(tf.float32, [None,10],name =networks[mm] +'y')
    
X = tf.placeholder(tf.float32, shape = [None, 32, 32, 3],name = 'x')
Dcon = tf.placeholder(tf.float32, shape = [None, 32, 32, 3],name = 'Dcon')
Y = tf.placeholder(tf.float32, [None,10],name ='y')
b = tf.placeholder(tf.bool,shape=(),name='b')
learning_rate = tf.placeholder(tf.float32,shape=(),name= 'learning_rate')
alpha = tf.placeholder(tf.float32,shape=(),name='alpha')
stop_grad = tf.placeholder(tf.bool,shape=(),name='stop')

x_list = []
y_list = []
output_list = []
Eent_noise = 0
output1 = 0
pp = 0
for i in range(len(networks)):
    x_image, y_label, output, probs,_,_= vgg16(networks[i],False,XX_list,YY_list,5e-4)
    _, output_noise = vgg16_1(Dcon,networks[i],False,stop_grad,XX_list,YY_list)
    _, _, output_train, probs_train,ent_train,loss= vgg16(networks[i],True,XX_list,YY_list,5e-4)
    x_list.append(x_image)
    y_list.append(y_label)
    output_list.append(output)
    pp+=loss
    Eent_noise += tf.reduce_sum(-tf.log(output_noise+1e-30)*output_noise,1)
    output1 += output_noise
    
output1 = output1/len(networks)
ent_noise = tf.reduce_sum(-tf.log(output1+1e-30)*output1,1)
Eent_noise = Eent_noise/len(networks)
MI = ent_noise # - Eent_noise
MI_mean=tf.reduce_mean(MI)
    
loss = pp-MI_mean*alpha
train_opt = tf.train.MomentumOptimizer(learning_rate,0.9,use_nesterov=True).minimize(loss)

saver = tf.train.Saver(max_to_keep = 10)
config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())
model_id=0
#save_path = saver.restore(sess, "./density_estimation_cifar10_A/classification_mnist666097/model.ckpt" )
epoch = 575
num_iter = epoch*180
for iter in range(num_iter):

    if iter < 80*epoch:
        a = 0.01
    elif iter < 120*epoch:
        a = 0.01*0.1
    elif iter < 160*epoch:
        a = 0.01*0.1*0.1
    else:
        a = 0.01*0.1*0.1*0.1

    if iter%1000 == 0:
        outputs=[]
        x_test_batch = np.copy(x_test)
        y_test_batch = np.copy(y_test)
        x_test_batch = stand(x_test_batch)
        for i in range(len(networks)):
            probs = []
            for j in range(10):   
              prob= sess.run(output_list[i], {x_list[i]:x_test_batch[j*1000:(j+1)*1000], y_list[i]:y_test_batch[j*1000:(j+1)*1000], b:False})
              probs.extend(prob)  
            outputs.append(probs)
        outputs=np.array(outputs) 
        
        accs = np.mean(np.argmax(outputs,2)==np.argmax(y_test_batch,1),1)
        print("accs: ",accs)

        softmax = np.mean(outputs,0)
        maxp_in = np.max(softmax,1)
        acc = np.mean(np.argmax(softmax,1) ==  np.argmax(y_test_batch,1))
        print( iter, 'Final Testing Accuracy: ', acc)
        
        ent_in = np.sum(-np.log(softmax+1e-11)*softmax,1)
        Eent_in = np.mean(np.sum(-np.log(outputs+1e-11)*outputs,2),0)
        MI_in = ent_in - Eent_in
        
        right = np.argmax(softmax,1) ==  np.argmax(y_test_batch,1)
        wrong_data0 = x_test_batch[~right]
        wrong_data = wrong_data0[np.random.randint(0,len(wrong_data0),10000)]
        right_data = x_test_batch[right]
        right_data = right_data[np.random.randint(0,len(right_data),10000)]
        
        outputs_right=[]
        for i in range(len(networks)):
            prob= sess.run(output_list[i], {x_list[i]:right_data, b:False})
            outputs_right.append(prob)
        outputs_right=np.array(outputs_right) 
        softmax_right = np.mean(outputs_right,0)
        maxp_in_right = np.max(softmax_right,1)
        ent_in_right = np.sum(-np.log(softmax_right+1e-11)*softmax_right,1)
        Eent_in_right = np.mean(np.sum(-np.log(outputs_right+1e-11)*outputs_right,2),0)
        MI_in_right = ent_in_right - Eent_in_right

        for t in range(0,8):
          if t == 0:
              safe_images = np.reshape(np.random.normal(0,1,[32*32*3*10000]),[10000,32,32,3])
              print("noise:")
          if t == 1:
              safe_images = TIM[np.random.randint(0,10000,10000)]
              print("TIM:")
          if t  == 2:
              safe_images = cifar10
              print("cifar10:") 
          if t  == 3:
              safe_images = LSUN
              print("LSUN:")          
          if t == 4:
              safe_images = LSUN
              maxp_in = maxp_in_right
              ent_in = ent_in_right
              MI_in = MI_in_right
              print("LSUN vs true:")  
          if t  == 5:
              safe_images = np.reshape(np.random.normal(0,1,[32*32*3*10000]),[10000,32,32,3])
              maxp_in = maxp_in_right
              ent_in = ent_in_right
              MI_in = MI_in_right
              print("gauss vs true:")
          if t == 6:
              safe_images = TIM[np.random.randint(0,10000,10000)]
              maxp_in = maxp_in_right
              ent_in = ent_in_right
              MI_in = MI_in_right
              print("TIM vs true:")
          if t  == 7:
              safe_images = cifar10
              maxp_in = maxp_in_right
              ent_in = ent_in_right
              MI_in = MI_in_right
              print("cifar10 vs true:")                    
                
          safe_images = stand(safe_images)   
    
          outputs_OOD=[]
          for j in range(len(networks)):
              probs_OOD  = []
              for r in range(10):
                prob_OOD = sess.run(output_list[j], {x_list[j]:safe_images[r*1000:(r+1)*1000], b:False})
                probs_OOD.extend(prob_OOD)  
              outputs_OOD.append(probs_OOD)
          outputs_OOD = np.array(outputs_OOD) 

          softmax_OOD = np.mean(outputs_OOD,0)
          maxp_OOD = np.max(softmax_OOD,1)
          ent_OOD = np.sum(-np.log(softmax_OOD+1e-11)*softmax_OOD,1)
          Eent_OOD = np.mean(np.sum(-np.log(outputs_OOD+1e-11)*outputs_OOD,2),0)
          MI_OOD = ent_OOD - Eent_OOD

          print("maxp_OOD:",np.mean(maxp_OOD>0.99),np.mean(maxp_OOD),np.std(maxp_OOD))
          print("maxp_inD:",np.mean(maxp_in>0.99),np.mean(maxp_in),np.std(maxp_in))
          print("ent_OOD:",np.mean(ent_OOD),np.std(ent_OOD), "ent_in:", np.mean(ent_in),np.std(ent_in))
          print("MI_OOD:",np.mean(MI_OOD),np.std(MI_OOD), "ent_in:", np.mean(MI_in),np.std(MI_in))

          safe, risky  = -np.reshape(maxp_in,[10000,1]), -np.reshape(maxp_OOD,[10000,1])
          labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
          labels[safe.shape[0]:] += 1
          examples = np.squeeze(np.vstack((safe, risky)))
          print('AUPR_p:', round(100*average_precision_score(labels, examples), 2))
          print('AUROC_p:', round(100*roc_auc_score(labels, examples), 2))
          print("FPR95:",ErrorRateAt95Recall1(labels, examples)*100)
        
          safe, risky = np.reshape(ent_in,[10000,1]), np.reshape(ent_OOD,[10000,1])
          labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
          labels[safe.shape[0]:] += 1
          examples = np.squeeze(np.vstack((safe, risky)))
          print('AUPR_entropy:', round(100*average_precision_score(labels, examples), 2))
          print('AUROC_entropy:', round(100*roc_auc_score(labels, examples), 2))
          print("FPR95:",ErrorRateAt95Recall1(labels, examples)*100)
        
          safe, risky = np.reshape(MI_in,[10000,1]), np.reshape(MI_OOD,[10000,1])
          labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
          labels[safe.shape[0]:] += 1
          examples = np.squeeze(np.vstack((safe, risky)))
          print('AUPR_MI:', round(100*average_precision_score(labels, examples), 2))
          print('AUROC_MI:', round(100*roc_auc_score(labels, examples), 2))
          print("FPR95:",ErrorRateAt95Recall1(labels, examples)*100)
          print("############################################")
        
    my_noise = np.copy(x_train[np.random.randint(0,50000,batch_size)])
    my_noise = crop(my_noise)
    my_noise = stand(my_noise)
    c = 0.5
    d = False
        
    x_batch_list = []
    y_batch_list = []
    for hh in range(len(networks)):
        seed = np.random.randint(0,50000,batch_size)
        x_batch = x_train[seed]
        y_batch = y_train[seed]
        x_batch = preprocess(x_batch) 
        x_batch_list.append(x_batch)
        y_batch_list.append(y_batch)     
    
    sess.run(train_opt, {x_list[0]:x_batch_list[0], y_list[0]:y_batch_list[0],
                         alpha:c, b:True, Dcon:my_noise, learning_rate:a,stop_grad:d})  
    if iter%1000==0:
        #save_path = saver.save(sess, "./density_estimation_cifar10_A/classification_mnist%s/model.ckpt" % model_id)
        print("model",model_id,"savedp")
        model_id+=1