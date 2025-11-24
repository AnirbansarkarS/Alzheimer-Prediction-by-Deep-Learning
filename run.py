import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import itertools


DATASET_DIR = "dataset/"   # change if needed
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15


# ---------------------------
# DATA PIPELINE
# ---------------------------

def load_data():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        subset='training'
    )

    valid = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    return train, valid


# ---------------------------
# BUILD MODEL (MobileNetV2)
# ---------------------------

def build_model(n_classes):
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    output = Dense(n_classes, activation="softmax")(x)

    model = Model(base.input, output)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ---------------------------
# TRAIN MODEL
# ---------------------------

def train_model():
    train, valid = load_data()
    model = build_model(n_classes=train.num_classes)

    callbacks = [
        ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    ]

    history = model.fit(
        train,
        validation_data=valid,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    return model, train, valid


# ---------------------------
# GRAD-CAM IMPLEMENTATION
# ---------------------------

def make_grad_cam(model, img_array, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)[0]
    heatmap /= np.max(heatmap)

    return heatmap, int(pred_index)


def save_grad_cam(model, image_path, save_path="gradcam_result.jpg"):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    heatmap, pred_class = make_grad_cam(model, img_array)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, superimposed)

    print("Grad-CAM saved:", save_path)



# ---------------------------
# ERROR ANALYSIS
# ---------------------------

def error_analysis(model, valid):
    valid.reset()
    preds = model.predict(valid)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = valid.classes

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=list(valid.class_indices.keys())))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)

    # plot confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = list(valid.class_indices.keys())
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.show()



# ---------------------------
# MAIN ENTRYPOINT
# ---------------------------

if __name__ == "__main__":
    print("Training model...")
    model, train, valid = train_model()

    print("\nRunning error analysis…")
    error_analysis(model, valid)

    print("\nSaving Grad-CAM for a sample image…")
    sample_image = os.path.join(DATASET_DIR, list(train.class_indices.keys())[0],
                                os.listdir(os.path.join(DATASET_DIR, list(train.class_indices.keys())[0]))[0])

    save_grad_cam(model, sample_image, "gradcam_sample.jpg")

    print("\nDONE ✔")
