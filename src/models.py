import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D,Activation,GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, MaxPool2D, Dense,Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers.experimental.preprocessing import Resizing
import tensorflow.keras.backend as K


# define twin discriminator

def mul_layer(x):
    y =  tf.math.scalar_mul(x[1], x[0])
    return y 
    
init = RandomNormal(stddev=0.02)

Conv1 = Conv2D(16, (7, 7),activation='relu',kernel_initializer=init) 
Conv2 = Conv2D(32, (5, 5),activation='relu',kernel_initializer=init) 
Conv3 = Conv2D(64, (3, 3),activation='relu',kernel_initializer=init) 
Conv4 = Conv2D(128, (3, 3),activation='relu',kernel_initializer=init) 

wt = 0.2

def twin_cnn(input_img_layer):
    
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
    
    merge_out = Concatenate(axis=1)([gap4, gap3, gap2])
    
    return merge_out

 
def twin_discriminator(input_shape):
 
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    feat_left = twin_cnn (left_input)
    feat_right = twin_cnn (right_input)
        
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([feat_left, feat_right])
        
    prediction = Dense(1,activation='sigmoid')(L1_distance)
 
    model = Model(inputs=[left_input,right_input],outputs=prediction)
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    
    return model

# define CNN discriminator

def cnn_discriminator(image_shape):
    
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
    
    d = Dense(1)(d)
    d = Activation('sigmoid')(d) 
    
    model = Model(in_src_image, d)
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    
    return model

# define the encoder block
def encoder_block(layer_in, n_filters, batchnorm=True):

 	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
     
 	if batchnorm:
         g = BatchNormalization()(g, training=True)
         
 	g = LeakyReLU(alpha=0.2)(g)
     
 	return g
 
# define the decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
        
    g = Resizing (skip_in.shape[1],skip_in.shape[2])(layer_in)
    
    g = Conv2D(n_filters, (4,4),   padding='same', kernel_initializer=init)(g)
        
    g = BatchNormalization()(g, training=True)
        
    if dropout:
        
        g = Dropout(0.5)(g, training=True)
            
    g = Concatenate()([g, skip_in])
        
    g = Activation('relu')(g)
    
    return g
     
   
# define generator 
def generator(image_shape):
        
    in_image = Input(shape=image_shape)
    
    # encoding path
    e1 = encoder_block(in_image, 32, batchnorm=False)
    e2 = encoder_block(e1, 64)
    e3 = encoder_block(e2, 128)
    e4 = encoder_block(e3, 256)
    e5 = encoder_block(e4, 256)

    # bottleneck 
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5)
    b = Activation('relu')(b)
    
    # decoding path
    d3 = decoder_block(b, e5, 256)
    d4 = decoder_block(d3, e4, 256, dropout=False)
    d5 = decoder_block(d4, e3, 128, dropout=False)
    d6 = decoder_block(d5, e2, 64, dropout=False)
    d7 = decoder_block(d6, e1, 32, dropout=False)
    
    # output
    g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    
    model = Model(in_image, out_image) 
    return model