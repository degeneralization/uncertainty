
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import os
import cv2



# get TinyImageNet

NUM_IMAGES = 100000
TRAINING_IMAGES_DIR = './tiny-imagenet-200/train/'
IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS
def load_training_images(image_dir, batch_size=500):

    image_index = 0
    images = np.ndarray(shape=(NUM_IMAGES, IMAGE_ARR_SIZE))
    names = []
    labels = []                       
    
    print("Loading training images from ", image_dir)
    # Loop through all the types directories
    for type in os.listdir(image_dir):
        if os.path.isdir(image_dir + type + '/images/'):

            type_images = os.listdir(image_dir + type + '/images/')
            # Loop through all the images of a type directory
            batch_index = 0;
            #print ("Loading Class ", type)
            for image in type_images:
                image_file = os.path.join(image_dir, type + '/images/', image)

                # reading the images as they are; no normalization, no color editing
                image_data = cv2.imread(image_file) 
                image_data = np.array(image_data)
                #print ('Loaded Image', image_file, image_data.shape)
                if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
                    images[image_index, :] = image_data.flatten()

                    labels.append(type)
                    names.append(image)
                    
                    image_index += 1
                    batch_index += 1
                if (batch_index >= batch_size):
                    break;
    
    print("Loaded Training Images", image_index)
    return (images, np.asarray(labels), np.asarray(names))

## sample 10000 data points as OOD data
data = load_training_images(TRAINING_IMAGES_DIR, batch_size=500)
seed = np.random.randint(0,100000,10000)
data = data[0][seed]
TIM = data
data = 1
TIM = np.reshape(TIM,[10000,64,64,3])/255.
TIM1=[]
for i in range(len(TIM)):
    im = cv2.resize(TIM[i], dsize=(32,32), interpolation=cv2.INTER_CUBIC)
    TIM1.append(im)
TIM1 = np.array(TIM1)
np.save("TIM",TIM1)
TIM1 = 1
#TIM = np.load("./TIM.npy")  

# get LSUN data
data, info = tfds.load("lsun", with_info=True)
train_data = data['train']
builder = tfds.builder("lsun")
builder.download_and_prepare()
datasets = builder.as_dataset()
np_datasets = tfds.as_numpy(datasets)
np_datasets

LSUN = []
for example in np_datasets["train"]:
    image = example['image']
    res = cv2.resize(image, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    LSUN.append(res)
LSUN = np.array(LSUN)
np.save("LSUN.npy",LSUN[:10000])
