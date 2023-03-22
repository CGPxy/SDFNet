#coding=utf-8
import matplotlib

import matplotlib.pyplot as plt
import argparse
import numpy as np  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import *
from tensorflow.python.layers import utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
import os
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.ndimage import distance_transform_edt as distance
from model_net import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

seed = 7 
np.random.seed(seed)  

matplotlib.use("Agg")
img_w = 384  
img_h = 384


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        
        #print(img.shape)

    else:
        # img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
        # print(img.shape)
    return img

filepath ='/media/dy/Data_2T/CGP/Unet_Segnet/data/kidney/new-kidney/1/Train_images/Augementa/'



def get_train_val(val_rate=0.25):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(filepath + 'Train_images/'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


# data for training
def generateData(batch_size, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'Train_images/' + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(filepath + 'Train_labels/' + url, grayscale=True)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                # yield (train_data, train_label)
                yield (train_data,{'out': train_label, 'Fc': train_label}) 
                train_data = []
                train_label = []
                batch = 0


# data for validation
def generateValidData(batch_size, data=[]):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'Train_images/' + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(filepath + 'Train_labels/' + url, grayscale=True)
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                # yield (valid_data, valid_label)
                yield (valid_data,{'out': valid_label, 'Fc': valid_label})
                valid_data = []
                valid_label = []
                batch = 0

"""
Binary form of focal loss.
"""
def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed

def edge_loss():
    def loss(y_true, y_pred):
        y_true_residual = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(y_true)
        y_true_edg = K.abs(Subtract()([y_true, y_true_residual]))

        y_pred_residual = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(y_pred)
        y_pred_edg = K.abs(Subtract()([y_pred, y_pred_residual]))

        return tf.keras.metrics.mean_squared_error(y_true_edg, y_pred_edg)
    return loss

def train(args): 
    EPOCHS = 50
    BS = 12
    model = SDFNet()
    model.summary() 

    # step1:lr=1e-3,step2:lr=1e-4,step3=1e-5 
    model.compile(loss={'out':binary_focal_loss(alpha=.25, gamma=2), 'Fc':edge_loss()}, loss_weights={'out': 1.0, 'Fc': 1.0}, metrics=["accuracy"], optimizer=Adam(lr=1e-3))
    checkpointer = ModelCheckpoint(os.path.join(
        args['save_dir'], 'model_{epoch:03d}.hdf5'), monitor='val_acc', save_best_only=False, mode='max')

    tensorboard = TensorBoard(log_dir='./logs/kidney/1/', histogram_freq=0, write_graph=True, write_images=True) 

    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=[checkpointer, tensorboard])  
  
def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--save_dir", default="/media/dy/Data_2T/CGP/Unet_Segnet/SDFNet/model/kidney/1/",
                    help="path to output model")
    args = vars(ap.parse_args()) 
    return args


if __name__=='__main__':  
    args = args_parse()
    train(args)  
    #predict()  
