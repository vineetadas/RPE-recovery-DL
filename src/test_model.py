from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D,Add
from tensorflow.keras.layers import Conv2DTranspose,RepeatVector, Permute, Multiply
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation,GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout, MaxPool2D, Flatten, Dense,Lambda, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from  matplotlib import pyplot  
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
import tensorflow as tf
import pickle
from PIL import Image
import os


####  Define Model


def mul_layer(x):
    y =  tf.math.scalar_mul(x[1], x[0])
    return y 


wt =0.2  
  
init = RandomNormal(stddev=0.02)

Conv1 = Conv2D(16, (7, 7),activation='relu',kernel_initializer=init) 
Conv2 = Conv2D(32, (5, 5),activation='relu',kernel_initializer=init) 
Conv3 = Conv2D(64, (3, 3),activation='relu',kernel_initializer=init) 
Conv4 = Conv2D(128, (3, 3),activation='relu',kernel_initializer=init) 

def dists_feature_extraction_model(input_img_layer):
    
    conv1 = Conv1 (input_img_layer)  
    gap1 = GlobalAveragePooling2D()(conv1)
    gap1 = Activation('sigmoid')(gap1)
    
    pool1  = MaxPool2D() (conv1)
    
    conv2 = Conv2 (pool1)
    gap2 = GlobalAveragePooling2D()(conv2)
    gap2 = Activation('sigmoid')(gap2)
    gap2 = Lambda(mul_layer) ([gap2, wt])
    
    pool2 = MaxPool2D()(conv2)

    
    conv3 = Conv3 (pool2) 
    gap3 = GlobalAveragePooling2D()(conv3)
    gap3 = Activation('sigmoid')(gap3)
    gap3 = Lambda(mul_layer) ([gap3, wt])
    
    pool3 = MaxPool2D() (conv3) 
    
    conv4 = Conv4 (pool3)
    gap4 = GlobalAveragePooling2D()  (conv4)
    gap4 = Activation('sigmoid')(gap4)
    # pool4 = MaxPool2D() (conv4_2)   
    
    merge_out = Concatenate(axis=1)([gap4, gap3, gap2])
    
    return merge_out

 
def def_similarity_model(input_shape):
 
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    feat_left = dists_feature_extraction_model (left_input)
    feat_right = dists_feature_extraction_model (right_input)
    
    # gap_left = Activation('sigmoid') (feat_left)
    # gap_right = Activation('sigmoid') (feat_right)
    
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([feat_left, feat_right])
    
    
    # dist_score = Lambda(score_computation) ([feat_left,feat_right])
    
    prediction = Dense(1,activation='sigmoid')(L1_distance)
 
    # Connect the inputs with the outputs
    similarity_net = Model(inputs=[left_input,right_input],outputs=prediction)
    similarity_net.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    # return the model
    return similarity_net

    
 # define the discriminator SRGAN model
def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.02) 
    in_src_image = Input(shape=image_shape)
    d = Conv2D(16, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_src_image)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(32, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = GlobalAveragePooling2D()(d)
    # d = LeakyReLU(alpha = 0.2)(d)
    d = Dense(1)(d)
    d = Activation('sigmoid')(d) 
    model = Model(in_src_image, d)
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
 	# weight initialization
 	init = RandomNormal(stddev=0.02)
 	# add downsampling layer
 	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
 	# conditionally add batch normalization
 	if batchnorm:
         g = BatchNormalization()(g, training=True)
 	# leaky relu activation
 	g = LeakyReLU(alpha=0.2)(g)
 	return g
 
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    
    init = RandomNormal(stddev=0.02)
    
    g = Resizing (skip_in.shape[1],skip_in.shape[2])(layer_in)
    
    g = Conv2D(n_filters, (4,4),   padding='same', kernel_initializer=init)(g)
    
    # add batch normalization
    
    g = BatchNormalization()(g, training=True)
    
    # conditionally add dropout
    
    if dropout:
        
        g = Dropout(0.5)(g, training=True)
        
    # merge with skip connection
    
    g = Concatenate()([g, skip_in])
    
    # relu activation
    
    g = Activation('relu')(g)
    
    return g
     
   
# define the standalone generator model
def define_generator(image_shape):
    
    init = RandomNormal(stddev=0.02) 
    
    in_image = Input(shape=image_shape)
     	# encoder model
    e1 = define_encoder_block(in_image, 32, batchnorm=False)
    e2 = define_encoder_block(e1, 64)
    e3 = define_encoder_block(e2, 128)
    e4 = define_encoder_block(e3, 256)
    e5 = define_encoder_block(e4, 256)
    # e6 = define_encoder_block(e5, 256)
    # e7 = define_encoder_block(e6, 256)
    # # 	# bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5)
    b = Activation('relu')(b)
    # # 	# decoder model
    # d1 = decoder_block(b, e7, 256)
    # d2 = decoder_block(b, e6, 256)
    d3 = decoder_block(b, e5, 256)
    d4 = decoder_block(d3, e4, 256, dropout=False)
    d5 = decoder_block(d4, e3, 128, dropout=False)
    d6 = decoder_block(d5, e2, 64, dropout=False)
    d7 = decoder_block(d6, e1, 32, dropout=False)
     	# output
    g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
     	# define model
    model = Model(in_image, out_image) 
    # model.summary()
    return model
 	

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, s_model, image_shape):
    # make weights in the discriminator not trainable
    
    s_model.trainable= False    
    d_model.trainable= False  
    # for layer in d_model.layers:
    #     if not isinstance(layer, BatchNormalization):
    #         layer.trainable = False
                
    # define the source image   
    in_src = Input(shape=image_shape)
    in_ref = Input(shape=image_shape)
    
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model(gen_out)
    
    s_out = s_model ([in_ref, gen_out])
    
    # src image as input, generated image and classification output
    
    model = Model([in_src, in_ref], [dis_out, s_out, gen_out])
    # compile model
    
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy','binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,1,100])
    
    return model

image_shape  = (200,300,1)

d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)

g_model.load_weights('D:/vineeta/fast_ao_oct_RPE/rpe_enhancement/gan_new/final/results_prop_method/new_modelWts_ab_weighted_concat_gap432_except1105_bsize_8_ep100_wt0p2/Weights_g_model_gan_51.h5',by_name=True)

def load_real_samples(X):
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X


train_datagen = ImageDataGenerator(preprocessing_function=load_real_samples)

#train_datagen = ImageDataGenerator(rescale=1/255)

batch_Size=1

train_generator_1 = train_datagen.flow_from_directory(
    directory="D:/vineeta/fast_ao_oct_RPE/rpe_enhancement/DATA/test_images/dewarped_img_NMI/2dPSD fig/speckled",
    target_size=(200, 300),
    batch_size=batch_Size,
    color_mode="grayscale",
    class_mode=None,
    shuffle=False,
    seed=42
    
) 
batch_count=train_generator_1.n // batch_Size 

fileNames=train_generator_1.filenames


rec_image_storage_path = 'D:/vineeta/fast_ao_oct_RPE/rpe_enhancement/DATA/test_images/dewarped_img_NMI/2dPSD fig'

for val in range(batch_count):
    temp_img=train_generator_1.next()
    nameF=fileNames[val] 
    gen_img = g_model.predict(temp_img)
    channel1=gen_img[0,:,:,0]
    channel1 = (channel1 - channel1.min()) / (channel1.max() - channel1.min())
    result = Image.fromarray((channel1 * 255).astype(np.uint8))
    result.save(os.path.join(rec_image_storage_path, nameF[2:len(nameF)]))

