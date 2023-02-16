import os

import numpy as np
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import generator, cnn_discriminator

IMAGE_SHAPE = (200, 300, 1)
PATH_TO_TEST_DATA = "./data/test_data/input"
PATH_TO_RESULTS = "./data/test_data/result"
PATH_TO_TRAINED_MODEL = "./data/trained_model/trained_model_weights.h5"


def load_real_samples(X):
    X = X.astype("float32")
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


if __name__ == "__main__":
    # loading the trained model
    d_model = cnn_discriminator(IMAGE_SHAPE)
    g_model = generator(IMAGE_SHAPE)
    g_model.load_weights(PATH_TO_TRAINED_MODEL, by_name=True)

    train_datagen = ImageDataGenerator(
        preprocessing_function=load_real_samples
    )

    train_generator_1 = train_datagen.flow_from_directory(
        directory=PATH_TO_TEST_DATA,
        target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]),
        batch_size=1,
        color_mode="grayscale",
        class_mode=None,
        shuffle=False,
        seed=42,
    )

    batch_count = train_generator_1.n

    fileNames = train_generator_1.filenames

    for val in range(batch_count):
        temp_img = train_generator_1.next()
        nameF = Path(fileNames[val]).name
        gen_img = g_model.predict(temp_img).squeeze()
        gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min())
        result = Image.fromarray((gen_img * 255).astype(np.uint8))
        result.save(os.path.join(PATH_TO_RESULTS, nameF))
