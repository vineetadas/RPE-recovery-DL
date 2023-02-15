from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os
from models import generator,cnn_discriminator

image_shape  = (200,300,1)
path_to_test_data = "./data/test_data/input"
path_to_results = './data/test_data/result'
path_to_trained_model = './data/trained_model/trained_model_weights.h5' 

# loading the trained model

d_model = cnn_discriminator(image_shape)

g_model = generator(image_shape)

g_model.load_weights(path_to_trained_model ,by_name=True)

# loading the test dataset
def load_real_samples(X):
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

train_datagen = ImageDataGenerator(preprocessing_function=load_real_samples)

train_generator_1 = train_datagen.flow_from_directory(
    directory = path_to_test_data,
    target_size = (image_shape[0],image_shape[1]),
    batch_size = 1,
    color_mode = "grayscale",
    class_mode = None,
    shuffle = False,
    seed = 42
    
) 

batch_count=train_generator_1.n 

fileNames=train_generator_1.filenames

for val in range(batch_count):
    temp_img=train_generator_1.next()
    nameF=fileNames[val].split('\\')[1]
    gen_img = g_model.predict(temp_img).squeeze()
    gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min())
    result = Image.fromarray((gen_img * 255).astype(np.uint8))
    result.save(os.path.join(path_to_results, nameF))

