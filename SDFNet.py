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

img_w = 384  
img_h = 384


def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat

def SNet(data):
    conv01 = Conv2D(16, (1, 1), padding="same")(data)
    batch01 = BatchNormalization()(conv01)
    LeakyReLU01 = LeakyReLU(alpha=0.01)(batch01)

    conv02 = Conv2D(32, (3, 3), padding="same")(data)
    batch02 = BatchNormalization()(conv02)
    LeakyReLU02 = LeakyReLU(alpha=0.01)(batch02)

    conv03 = Conv2D(16, (5, 5), padding="same")(data)
    batch03 = BatchNormalization()(conv03)
    LeakyReLU03 = LeakyReLU(alpha=0.01)(batch03)

    concatenate1 = concatenate([LeakyReLU01, LeakyReLU02], axis=-1)
    concatenate2 = concatenate([concatenate1, LeakyReLU03], axis=-1)
  
    conv04 = Conv2D(64, (3, 3), dilation_rate=2, padding="same")(concatenate2)
    batch04 = BatchNormalization()(conv04)
    LeakyReLU04 = LeakyReLU(alpha=0.01, name='Fs')(batch04)
    
    dropout = Dropout(0.2)(LeakyReLU04)
    dropout = Conv2D(1, (1, 1), activation='sigmoid',name='Fc')(dropout)

    return LeakyReLU04, dropout
  
def SDFNet():  
      
    inputs = Input((img_h,img_w, 3))
    batch = BatchNormalization(axis=-1)(inputs)

    FS, FC = SNet(data=batch)

    concatenate1 = concatenate([batch, FS], axis=-1)
    conv1 = Conv2D(64, (3, 3), padding="same")(concatenate1)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    conv2 = Conv2D(64, (3, 3), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = LeakyReLU(alpha=0.01)(batch2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(LeakyReLU2)
    
    
    conv3 = Conv2D(128, (3, 3), padding="same")(pool1)
    batch3 = BatchNormalization()(conv3)
    LeakyReLU3 = LeakyReLU(alpha=0.01)(batch3)
    conv4 = Conv2D(128, (3, 3), padding="same")(LeakyReLU3)
    batch4 = BatchNormalization()(conv4)
    LeakyReLU4 = LeakyReLU(alpha=0.01)(batch4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(LeakyReLU4)
    
    
    conv5 = Conv2D(256, (3, 3), padding="same")(pool2)
    batch5 = BatchNormalization()(conv5)
    LeakyReLU5 = LeakyReLU(alpha=0.01)(batch5)
    conv6 = Conv2D(256, (3, 3), padding="same")(LeakyReLU5)
    batch6 = BatchNormalization()(conv6)
    LeakyReLU6 = LeakyReLU(alpha=0.01)(batch6)
    conv7 = Conv2D(256, (3, 3), padding="same")(LeakyReLU6)
    batch7 = BatchNormalization()(conv7)
    LeakyReLU7 = LeakyReLU(alpha=0.01)(batch7)
    pool3 = MaxPooling2D(pool_size=(2, 2))(LeakyReLU7)
    
    conv8 = Conv2D(512, (3, 3), padding="same")(pool3)
    batch8 = BatchNormalization()(conv8)
    LeakyReLU8 = LeakyReLU(alpha=0.01)(batch8)
    conv9 = Conv2D(512, (3, 3), padding="same")(LeakyReLU8)
    batch9 = BatchNormalization()(conv9)
    LeakyReLU9 = LeakyReLU(alpha=0.01)(batch9)
    conv10 = Conv2D(512, (3, 3), padding="same")(LeakyReLU9)
    batch10 = BatchNormalization()(conv10)
    LeakyReLU10 = LeakyReLU(alpha=0.01)(batch10)
    pool4 = MaxPooling2D(pool_size=(2, 2))(LeakyReLU10)


    conv11 = Conv2D(512, (3, 3), padding="same")(pool4)
    batch11 = BatchNormalization()(conv11)
    LeakyReLU11 = LeakyReLU(alpha=0.01)(batch11)
    conv12 = Conv2D(512, (1, 1), padding="same")(LeakyReLU11)
    batch12 = BatchNormalization()(conv12)
    LeakyReLU12 = LeakyReLU(alpha=0.01)(batch12)
    conv13 = Conv2D(512, (1, 1), padding="same")(LeakyReLU12)
    batch13 = BatchNormalization()(conv13)
    LeakyReLU13 = LeakyReLU(alpha=0.01)(batch13)
    

## decode
    uppool1 = UpSampling2D((2, 2))(LeakyReLU13)
    upconv1 = Conv2D(512, (3, 3), padding="same")(uppool1)
    upbatch1 = BatchNormalization()(upconv1)
    upLeakyReLU1 = LeakyReLU(alpha=0.01)(upbatch1)
    upconv2 = Conv2D(512, (3, 3), padding="same")(upLeakyReLU1)
    upbatch2 = BatchNormalization()(upconv2)
    upLeakyReLU2 = LeakyReLU(alpha=0.01)(upbatch2)
    upconv3 = Conv2D(512, (3, 3), padding="same")(upLeakyReLU2)
    upbatch3 = BatchNormalization()(upconv3)
    upLeakyReLU3 = LeakyReLU(alpha=0.01)(upbatch3)
    
    
    uppool2 = UpSampling2D((2, 2))(upLeakyReLU3)
    upconv4 = Conv2D(256, (3, 3), padding="same")(uppool2)
    upbatch4 = BatchNormalization()(upconv4)
    upLeakyReLU4 = LeakyReLU(alpha=0.01)(upbatch4)
    upconv5 = Conv2D(256, (3, 3), padding="same")(upLeakyReLU4)
    upbatch5 = BatchNormalization()(upconv5)
    upLeakyReLU5 = LeakyReLU(alpha=0.01)(upbatch5)
    upconv6 = Conv2D(256, (3, 3), padding="same")(upLeakyReLU5)
    upbatch6 = BatchNormalization()(upconv6)
    upLeakyReLU6 = LeakyReLU(alpha=0.01)(upbatch6)
    


    uppool3 = UpSampling2D((2, 2))(upLeakyReLU6)
    upconv7 = Conv2D(128, (3, 3), padding="same")(uppool3)
    upbatch7 = BatchNormalization()(upconv7)
    upLeakyReLU7 = LeakyReLU(alpha=0.01)(upbatch7)
    upconv8 = Conv2D(128, (3, 3), padding="same")(upLeakyReLU7)
    upbatch8 = BatchNormalization()(upconv8)
    upLeakyReLU8 = LeakyReLU(alpha=0.01)(upbatch8)


    uppool4 = UpSampling2D((2, 2))(upLeakyReLU8)

    FC1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(FC)
    FC1 = K.abs(Subtract()([FC, FC1]))
    FC1 = expend_as(FC1, 64)

    concatenate2 = concatenate([uppool4, FC1], axis=-1)
    upconv9 = Conv2D(64, (1, 1), padding="same")(concatenate2)
    upbatch9 = BatchNormalization()(upconv9)
    upLeakyReLU9 = LeakyReLU(alpha=0.01)(upbatch9)
    upconv10 = Conv2D(64, (3, 3), padding="same")(concatenate2)
    upbatch10 = BatchNormalization()(upconv10)
    upLeakyReLU10 = LeakyReLU(alpha=0.01)(upbatch10)
    upconv11 = Conv2D(64, (5, 5), padding="same")(concatenate2)
    upbatch11 = BatchNormalization()(upconv11)
    upLeakyReLU11 = LeakyReLU(alpha=0.01)(upbatch11)
    concatenate3 = concatenate([upLeakyReLU9, upLeakyReLU10], axis=-1)
    concatenate4 = concatenate([concatenate3, upLeakyReLU11], axis=-1)

    outconv04 = Conv2D(1, (1, 1), activation='sigmoid',name='out')(concatenate4)

    model = Model(inputs=inputs, outputs=[outconv04, FC])
    return model