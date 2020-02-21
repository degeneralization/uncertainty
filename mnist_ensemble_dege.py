import tensorflow as tf
import numpy as np
import os
import scipy.io
import random
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import cv2
from utilities import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# omnilog data
import scipy.io as sio
import scipy.misc as scimisc
safe_list = [0,2,5,6,8,12,13,14,15,16,17,18,19,21,26]
m = sio.loadmat("./data/data_background.mat")
squished_set = []
for safe_number in safe_list:
    for alphabet in m['images'][safe_number]:
        for letters in alphabet:
            for letter in letters:
                for example in letter:
                    squished_set.append(scimisc.imresize(1 - example[0], (28,28)).reshape(1, 28*28))
omnilog = np.concatenate(squished_set, axis=0)
omnilog = np.reshape(omnilog,(9200,28,28,1))

# notMNIST data
img_size = 28
img_flat_size = img_size * img_size
folders_notMNIST = os.listdir('./data/notMNIST_small')
NotMNIST_x_list = []
for idx, folder in enumerate(folders_notMNIST):
    files_notMNIST = os.listdir('./data/notMNIST_small/' + folder)
    for file in files_notMNIST:
        img_NotMNIST = scipy.misc.imread('./data/notMNIST_small/' + folder + '/' + file, 0)
        NotMNIST_flat = np.reshape(img_NotMNIST, (img_flat_size))
        NotMNIST_x_list.append(NotMNIST_flat)        
NotMNIST_x = np.stack(NotMNIST_x_list, axis = 0)
NotMNIST = NotMNIST_x/255.
NotMNIST = np.reshape(NotMNIST,(len(NotMNIST),28,28,1))

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train = x_train/255.
x_test = x_test/255.
x_train = np.reshape(x_train,(60000,28,28,1))
x_test = np.reshape(x_test,(10000,28,28,1))
y_train = np.reshape(y_train,[60000,])
y_test = np.reshape(y_test,[10000,])
nb_classes = 10
targets = y_train.reshape(-1)
y_train = np.eye(nb_classes)[targets]
targets = y_test.reshape(-1)
y_test = np.eye(nb_classes)[targets]

batch_size = 128
tf.reset_default_graph()
networks = ['network1', 'network2', 'network3', 'network4', 'network5','network6', 'network7', 'network8', 'network9', 'network10']

XX_list={}
YY_list={}
for mm in range(len(networks)):
  with tf.name_scope(networks[mm]):
    XX_list[networks[mm]] = tf.placeholder(tf.float32, shape = [None, 28, 28, 1],name = networks[mm] +'x')
    YY_list[networks[mm]] = tf.placeholder(tf.float32, [None,10],name =networks[mm] +'y')
    
X = tf.placeholder(tf.float32, shape = [None, 28, 28, 1],name = 'x')
Dcon = tf.placeholder(tf.float32, shape = [None, 28, 28, 1],name = 'Dcon')
Y = tf.placeholder(tf.float32, [None,10],name ='y')
b = tf.placeholder(tf.bool,shape=(),name='b')
learning_rate = tf.placeholder(tf.float32,shape=(),name= 'learning_rate')
probability = tf.placeholder(tf.float32,shape=(),name='probability')
alpha = tf.placeholder(tf.float32,shape=(),name='alpha')
stop_grad = tf.placeholder(tf.bool,shape=(),name='stop')

x_list = []
y_list = []
output_list = []
Eent_noise = 0
output1 = 0
pp = 0
for i in range(len(networks)):
    x_image, y_label, output, probs,_,_= vgg4(networks[i],False,XX_list,YY_list,1e-6)
    _, output_noise = vgg4_1(Dcon,networks[i],False,stop_grad,XX_list,YY_list)
    _, _, output_train, probs_train,ent_train,loss= vgg4(networks[i],True,XX_list,YY_list,1e-6)
    x_list.append(x_image)
    y_list.append(y_label)
    output_list.append(output)
    pp+=loss
    Eent_noise += tf.reduce_sum(-tf.log(output_noise+1e-30)*output_noise,1)
    output1 += output_noise
    
output1 = output1/len(networks)
ent_noise = tf.reduce_sum(-tf.log(output1+1e-30)*output1,1)
Eent_noise = Eent_noise/len(networks)
MI = ent_noise - Eent_noise
MI_mean=tf.reduce_mean(MI)
    
pp = pp/len(networks)
loss = pp-MI_mean*alpha
train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver(max_to_keep = 10)
config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())
model_id=0
#save_path = saver.restore(sess, "./density_estimation_cifar10_A/classification_mnist666097/model.ckpt" )
epoch = 469
num_iter = epoch*25
for iter in range(num_iter):

    if iter < 1000:
        a = 0.0003
    elif iter < 5000:
        a = 0.0001
    elif iter < 10000:
        a = 0.00003
    elif iter < 20000:
        a = 0.00001
    else:
        a = 0.000001

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
        
        if iter%7 == 4 or iter%7 == 2 or iter%7 == 6:
            tick=0
            c = 3.
            d = False
        if iter%7 == 5:
            tick=1
            c = 3.
            d = True
        if iter%7 == 3 or iter%7 == 1 or iter%7 == 0:
            tick=0
            c = 3.
            d = False

        for t in range(0,6):
          if t == 1:
              safe_images = NotMNIST[np.random.randint(0,len(NotMNIST),10000)]
              print("notMNIST:")
          if t == 0:
              safe_images = np.reshape(np.random.normal(0,1,[28*28*1*10000]),[10000,28,28,1])
              print("noise:")
          if t  == 2:
              safe_images = omnilog[np.random.randint(0,len(omnilog),10000)]
              print("omnilog:")           
          if t  == 3:
              safe_images = omnilog[np.random.randint(0,len(omnilog),10000)]
              maxp_in = maxp_in_right
              ent_in = ent_in_right
              MI_in = MI_in_right
              print("omniglot vs true:")
          if t  == 4:
              safe_images = NotMNIST[np.random.randint(0,len(NotMNIST),10000)]
              maxp_in = maxp_in_right
              ent_in = ent_in_right
              MI_in = MI_in_right
              print("notmnist vs true:")           
          if t  == 5:
              safe_images = np.reshape(np.random.normal(0,1,[28*28*1*10000]),[10000,28,28,1])
              maxp_in = maxp_in_right
              ent_in = ent_in_right
              MI_in = MI_in_right
              print("gauss vs true:")                    
                
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
        
    if tick == 0 and iter%1==0:
        my_noise = np.copy(x_train[np.random.randint(0,50000,batch_size)])
        my_noise = crop_for_mnist(my_noise)
        my_noise = stand(my_noise)
    if tick == 1 and iter%100==0:
        my_noise = np.copy(x_train[np.random.randint(0,50000,batch_size)])
        my_noise = preprocess_for_mnist(my_noise)
        
    x_batch_list = []
    y_batch_list = []
    for hh in range(len(networks)):
        seed = np.random.randint(0,50000,batch_size)
        x_batch = x_train[seed]
        y_batch = y_train[seed]
        x_batch = preprocess_for_mnist(x_batch) 
        x_batch_list.append(x_batch)
        y_batch_list.append(y_batch)     
    
    sess.run(train_opt, {x_list[0]:x_batch_list[0], y_list[0]:y_batch_list[0],
                         x_list[1]:x_batch_list[1], y_list[1]:y_batch_list[1],
                         x_list[2]:x_batch_list[2], y_list[2]:y_batch_list[2],
                         x_list[3]:x_batch_list[3], y_list[3]:y_batch_list[3],
                         x_list[4]:x_batch_list[4], y_list[4]:y_batch_list[4],
                         x_list[5]:x_batch_list[5], y_list[5]:y_batch_list[5],
                         x_list[6]:x_batch_list[6], y_list[6]:y_batch_list[6],
                         x_list[7]:x_batch_list[7], y_list[7]:y_batch_list[7],
                         x_list[8]:x_batch_list[8], y_list[8]:y_batch_list[8],
                         x_list[9]:x_batch_list[9], y_list[9]:y_batch_list[9], 
                         alpha:c, b:True, Dcon:my_noise, learning_rate:a,stop_grad:d})  
    if iter%1000==0:
        #save_path = saver.save(sess, "./density_estimation_cifar10_A/classification_mnist%s/model.ckpt" % model_id)
        print("model",model_id,"savedp")
        model_id+=1