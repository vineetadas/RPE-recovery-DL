import os

from matplotlib import pyplot
import numpy as np
import pickle
from sklearn.utils import shuffle
from skimage import io
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from models import generator, twin_discriminator, cnn_discriminator

BATCH_SIZE = 8

# define the combined generator and discriminator model,
# for updating the generator
def define_gan(g_model, d_model, t_model, image_shape):
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
    gan_model,
    batch_create,
    batch_per_epoch,
    average_img_path,
    speckled_img_path,
    n_epochs=100,
):
    bat_per_epo = int(
        len(train_data_fname) / BATCH_SIZE
    )  # calculate the number of batches per training epoch

    for i in range(n_epochs):
        data_fname = shuffle(train_data_fname, random_state=i)
        for j in range(batch_per_epoch):
            rand_idx = batch_create[j]
            rand_idx = rand_idx.tolist()
            X_realA = get_batch_data(train_data_path_1, data_fname, rand_idx)
            X_realB = get_batch_data(train_data_path_120, data_fname, rand_idx)
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
            g_loss, _, _, _ = gan_model.train_on_batch(
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

    gan_model = define_gan(g_model, d_model, t_model, image_shape)

    # train model
    train(d_model, t_model, g_model, gan_model)
