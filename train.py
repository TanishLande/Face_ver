import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Layer
from tensorflow.keras.models import Model

# calculation metrics
from tensorflow.keras.metrics import Precision, Recall


POS_PATH = os.path.join("data", "positive")
NEG_PATH = os.path.join("data", "negatives")  # Fixed typo
ANC_PATH = os.path.join("data", "anchor")

# Device info (similar to torch.cuda.is_available())
# gpus = tf.config.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


anchor = tf.data.Dataset.list_files(ANC_PATH+"/*.jpg").take(250)
positive = tf.data.Dataset.list_files(POS_PATH+"/*.jpg").take(250)  # Fixed typo
negative = tf.data.Dataset.list_files(NEG_PATH+"/*.jpg").take(250)

def preprocess(file_path):  # Fixed typo in function name
    byte_path = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_path)
    img = tf.image.resize(img, (100,100))
    img = img / 255.0   # Fixed: was 225.0, should be 255.0
    return img 

positive_pairs = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negative_pairs = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positive_pairs.concatenate(negative_pairs)

def assign_preprocess(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)  # Fixed function name

data = data.map(assign_preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Training partition 
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Test data 
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

def embedding_layer():
    inp = Input(shape=(100,100,3), name="input_image")

    # First block 
    c1 = Conv2D(64, (10,10), activation="relu")(inp)
    m1 = MaxPool2D(64, (2,2), padding="same")(c1)
    
    # Second block 
    c2 = Conv2D(128, (7,7), activation="relu")(m1)
    m2 = MaxPool2D(64, (2,2), padding="same")(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation="relu")(m2)
    m3 = MaxPool2D(64, (2,2), padding="same")(c3)

    # Final block 
    c4 = Conv2D(256, (4,4), activation="relu")(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation="sigmoid")(f1)

    return Model(inputs=inp, outputs=d1, name="embedding")

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Making model
embedding_model = embedding_layer()

def make_siamese_model():
    input_image = Input(name="input_img", shape=(100,100,3))
    validation_image = Input(name="validation_img", shape=(100,100,3))

    siamese_layer = L1Dist()
    distances = siamese_layer(embedding_model(input_image), embedding_model(validation_image))
    classifier = Dense(1, activation="sigmoid")(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name="SiameseNetwork")

siamese_model = make_siamese_model()

binary_cross_loss = keras.losses.BinaryCrossentropy()
opt = keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        Y = batch[2]

        Y_pred = siamese_model(X, training=True)
        loss = binary_cross_loss(Y, Y_pred)

    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    return loss  # Added missing return statement

def train(data, EPOCHS):
    for epoch in range(1, EPOCHS+1):
        print("Epoch {}/{}".format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        for idx, batch in enumerate(data):
            loss = train_step(batch)
            progbar.update(idx+1, [("loss", loss)])

        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 50
train(train_data, EPOCHS)


test_input, test_val, y_pred = test_data.as_numpy_iterator().next()

y_hat = siamese_model.predict([test_input, test_val])


res = [ 1 if prediction > 0.5 else 0 for prediction in y_hat ]

# recall 
m = Recall()
m.update_state(y_pred, res)
m.result().numpy()

# precision
m = Recall()
m.update_state(y_pred, res)
m.result().numpy()

# save a model
siamese_model.save("siamesemodelv1.h5")

# load a model
new_model = tf.keras.models.load_model("siamesemodelv1.h5", custom_objects={"L1Dist":L1Dist, "BinaryCrossentropy":keras.losses.BinaryCrossentropy})

new_model.predict([test_input, test_val])