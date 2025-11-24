import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224

def ensure_channel(img):
    """
    Ensures grayscale -> (H,W,1)
    """
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    return img

def preprocess_for_tl(img):
    """
    Preprocess for MobileNetV2:
    - grayscale → RGB
    - resize
    - MobileNet preprocess_input
    """
    img = ensure_channel(img)
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    img = tf.image.grayscale_to_rgb(img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)

    return img.numpy()
