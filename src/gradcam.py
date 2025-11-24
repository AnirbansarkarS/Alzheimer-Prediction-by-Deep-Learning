import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- Grad-CAM Core Functions ----------

def get_last_conv_layer(model):
    """
    Auto-detects last convolution layer
    """
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:   # (H,W,C)
            return layer.name
    raise ValueError("No conv layer found")


def make_gradcam_heatmap(img_array, model):
    """
    Computes Grad-CAM heatmap
    """
    last_conv = get_last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tf.expand_dims(img_array, axis=0))
        class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap = heatmap / np.max(heatmap + 1e-9)

    return heatmap


def overlay_heatmap_on_image(original_img, heatmap, alpha=0.4):
    """
    Overlay CAM on grayscale MRI
    """
    heatmap_color = plt.cm.jet(heatmap)[..., :3]  # apply jet colormap
    heatmap_color = (heatmap_color * 255).astype(np.uint8)

    original_img_rgb = tf.image.grayscale_to_rgb(original_img)
    original_img_rgb = tf.image.resize(original_img_rgb, (224,224))

    original_np = original_img_rgb.numpy().astype("uint8")
    blended = (heatmap_color * alpha + original_np * (1 - alpha)).astype("uint8")

    return blended


# ---------- Save Correct/Wrong Grad-CAM ----------

def save_gradcam_examples(model, X, original_X, y_true, y_pred, class_names):
    if not os.path.exists("outputs/gradcam_correct"):
        os.makedirs("outputs/gradcam_correct")

    if not os.path.exists("outputs/gradcam_wrong"):
        os.makedirs("outputs/gradcam_wrong")

    for i in range(10):   # first 10 samples
        img = X[i]
        orig = original_X[i]
        heatmap = make_gradcam_heatmap(img, model)
        overlay = overlay_heatmap_on_image(orig, heatmap)

        file = f"{i}_{class_names[y_true[i]]}_pred-{class_names[y_pred[i]]}.png"
        path = "outputs/gradcam_correct" if y_pred[i] == y_true[i] else "outputs/gradcam_wrong"

        plt.imsave(f"{path}/{file}", overlay)
