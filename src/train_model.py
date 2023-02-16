from matplotlib import pyplot
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import generator, twin_discriminator, cnn_discriminator

 
# define the combined generator and discriminator model,
# for updating the generator
def define_pgan(g_model, d_model, t_model, image_shape):
    t_model.trainable = False
    d_model.trainable = False

    in_src = Input(shape=image_shape)
    in_ref = Input(shape=image_shape)

    gen_out = g_model(in_src)
    dis_out = d_model(gen_out)
    t_out = t_model([in_ref, gen_out])

    model = Model([in_src, in_ref], [dis_out, t_out, gen_out])

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(
        loss=["binary_crossentropy", "binary_crossentropy", "mae"],
        optimizer=opt,
        loss_weights=[1, 1, 100],
    )

    return model


def summarize_performance(step, g_model, X_realA, X_realB, log_dir):
    X_fakeB = g_model(X_realA)
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0

    pyplot.subplot(1, 3, 1)
    pyplot.axis("off")
    pyplot.imshow(X_realA[0, :, :, 0], cmap="gray")
    pyplot.subplot(1, 3, 2)
    pyplot.axis("off")
    pyplot.imshow(X_realB[0, :, :, 0], cmap="gray")
    pyplot.subplot(1, 3, 3)
    pyplot.axis("off")
    pyplot.imshow(X_fakeB[0, :, :, 0], cmap="gray")

    filename1 = log_dir + f"/epoch_images/plot_{step:06d}.png"

    pyplot.savefig(filename1)
    pyplot.close()


def save_model(g_model, ep, log_dir):
    g_model.save_weights(log_dir + f"/model/weights_g_model_pgan_{ep:d}.h5")


def load_real_samples(X):
    X = X.astype("float32")

    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


def train(
    d_model,
    s_model,
    g_model,
    pgan_model,
    ground_truth_img_path,
    input_img_path,
    batch_size=8,
    training_img_size=(150,150),
    n_epochs=100,
):
    """
    Parameters
    ----------
    d_model : The CNN discriminator network
    s_model : Twin discriminator network
    g_model : Generator network 
    pgan_model : P-GAN network
    ground_truth_img_path : path to the training ground truth images
    input_img_path : path to the input training inmages to the netwrok
    batch_size : Integer, The default is 8.
    training_img_size : 2D tensor with shape (width, height) of the training images.
    The default is (150,150).
    n_epochs : Integer, number of epochs to train the network.
     The default is 100.

    Returns
    -------
    None.

    """
    train_datagen = ImageDataGenerator(preprocessing_function=load_real_samples)

    train_generator_gt = train_datagen.flow_from_directory(
        directory = ground_truth_img_path,
        target_size = training_img_size,
        batch_size = batch_size,
        color_mode = "grayscale",
        class_mode = None,
        shuffle = False,
        seed = 42
        
    ) 

    train_generator_input = train_datagen.flow_from_directory(
        directory = input_img_path,
        target_size = training_img_size,
        batch_size = batch_size,
        color_mode = "grayscale",
        class_mode = None,
        shuffle = False,
        seed = 42
        
    ) 

    for i in range(n_epochs):
        batch_per_epoch = int(train_generator_gt.n/batch_size)
        for j in range(batch_per_epoch):
             
            X_realA = train_generator_input.next()
            X_realB = train_generator_gt.next()
            y_real = (
                np.ones(len(X_realA))
                - np.random.random_sample(len(X_realA)) * 0.2
            )
            y_real_s = (
                np.ones(len(X_realA))
                - np.random.random_sample(len(X_realA)) * 0.2
            )
            X_fakeB = g_model.predict(X_realA)
            y_fake = np.random.random_sample(len(X_fakeB)) * 0.2
            y_fake_s = np.random.random_sample(len(X_fakeB)) * 0.2

            if j % 3 == 0:
                # update CNN discriminator
                d_model.trainable = True
                d_loss1 = d_model.train_on_batch(X_realB, y_real)
                # update discriminator for generated samples
                d_loss2 = d_model.train_on_batch(X_fakeB, y_fake)
                d_model.trainable = False

                ## update twin discriminator
                s_model.trainable = True
                s_loss1 = s_model.train_on_batch([X_realB, X_realB], y_real_s)
                s_loss2 = s_model.train_on_batch([X_realB, X_fakeB], y_fake_s)
                s_model.trainable = False

            # update  generator
            g_loss, _, _, _ = pgan_model.train_on_batch(
                [X_realA, X_realB], [y_real, y_real_s, X_realB]
            )

            # summarize performance

            print(
                ">%d, d1[%.3f] d2[%.3f] s1[%.3f] s2[%.3f] g[%.3f]"
                % (i + 1, d_loss1, d_loss2, s_loss1, s_loss2, g_loss)
            )

        summarize_performance(i, g_model, X_realA, X_realB)
        save_model(g_model, s_model, i)


if __name__ == "__main__":
    image_shape = (150, 150, 1)

    d_model = cnn_discriminator(image_shape)
    g_model = generator(image_shape)
    t_model = twin_discriminator(image_shape)
    pgan_model = define_pgan(g_model, d_model, t_model, image_shape)
    
    """
    Provide path to the training data. The paths given below are examples. They do not contain training data
    """
    ground_truth_img_path = "./data/train_data/ground_truth"
    input_img_path = "./data/train_data/input"
    

    # train model
    train(d_model, t_model, g_model, pgan_model, ground_truth_img_path, input_img_path)
