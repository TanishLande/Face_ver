import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall
import cv2

# Disable GPU (same as training)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Data paths
POS_PATH = os.path.join("data", "positive")
NEG_PATH = os.path.join("data", "negatives")
ANC_PATH = os.path.join("data", "anchor")

# Custom L1 Distance layer (must be same as training)
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

def preprocess(file_path):
    """Same preprocessing function as training"""
    byte_path = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_path)
    img = tf.image.resize(img, (100,100))
    img = img / 255.0
    return img 

def prepare_test_data():
    """Prepare test data exactly like in training"""
    print("Loading test data...")
    
    anchor = tf.data.Dataset.list_files(ANC_PATH+"/*.jpg").take(250)
    positive = tf.data.Dataset.list_files(POS_PATH+"/*.jpg").take(250)
    negative = tf.data.Dataset.list_files(NEG_PATH+"/*.jpg").take(250)

    def assign_preprocess(input_img, validation_img, label):
        return (preprocess(input_img), preprocess(validation_img), label)

    positive_pairs = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negative_pairs = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positive_pairs.concatenate(negative_pairs)
    
    data = data.map(assign_preprocess)
    data = data.cache()
    data = data.shuffle(buffer_size=1024)

    # Get test data (same split as training)
    test_data = data.skip(round(len(data)*.7))
    test_data = test_data.take(round(len(data)*.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)
    
    return test_data

def create_model_architecture():
    """Recreate the exact model architecture from training"""
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

    def make_siamese_model():
        input_image = Input(name="input_img", shape=(100,100,3))
        validation_image = Input(name="validation_img", shape=(100,100,3))

        embedding_model = embedding_layer()
        siamese_layer = L1Dist()
        distances = siamese_layer(embedding_model(input_image), embedding_model(validation_image))
        classifier = Dense(1, activation="sigmoid")(distances)

        return Model(inputs=[input_image, validation_image], outputs=classifier, name="SiameseNetwork")

    return make_siamese_model()

def load_checkpoint_5():
    """Load specific checkpoint ckpt-5"""
    print("Loading checkpoint ckpt-5...")
    
    # Create model architecture
    siamese_model = create_model_architecture()
    opt = keras.optimizers.Adam(1e-4)
    
    # Create checkpoint object
    checkpoint_dir = './training_checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, "ckpt-5")
    
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
    
    # Load the specific checkpoint
    try:
        checkpoint.restore(checkpoint_path)
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
        return siamese_model
    except Exception as e:
        print(f"Error loading checqkpoint: {e}")
        return None

def verify_model(model, detection_threshold=0.5, verification_threshold=0.8):
    results = []
    input_img = preprocess(os.path.join("application_data", "input_images", "input_image.jpg"))
    input_img = np.expand_dims(input_img, axis=0)  # batch of 1

    for image in os.listdir(os.path.join("application_data", "verification_images")):
        validation_image = preprocess(os.path.join("application_data", "verification_images", image))
        validation_image = np.expand_dims(validation_image, axis=0)  # batch of 1

        # predict on the pair correctly
        result = model.predict([input_img, validation_image])
        results.append(result)

    # Convert to numpy array
    results = np.array(results).flatten()

    # Count detections above threshold
    detection = np.sum(results > detection_threshold)
    verification = detection / len(results)
    verified = verification > verification_threshold

    return verified, results



siamese_model = load_checkpoint_5()

# real time frame capture 
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret , frame = cap.read()

    frame = frame[120:120+250, 200:200+250, :]

    cv2.imshow("Verification", frame)

    if cv2.waitKey(10) & 0xFF==ord('v'):
        cv2.imwrite(os.path.join("application_data", "input_images", "input_image.jpg"), frame)

        # verification process
        model = load_checkpoint_5()
        verified, results = verify_model(siamese_model, 0.9, 0.7)
        print(verified)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()