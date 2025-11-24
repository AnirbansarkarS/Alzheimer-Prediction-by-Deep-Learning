**MRI Classification with MobileNetV2 + Explainability**

## 🧠 Project Overview

This project builds a **deep-learning pipeline** to classify MRI images using **MobileNetV2** with transfer learning. It includes:

* Grayscale → RGB preprocessing
* Data augmentation
* Transfer learning with MobileNetV2
* Evaluation (ROC, F1-score, Confusion Matrix)
* **Explainability using Grad-CAM heatmaps**
* **Error analysis** (class imbalance, misclassified samples, model weaknesses)

This repository is structured following a weekly learning roadmap (Week 0 → Week 3), inspired by real-world ML workflows.

---

## 🚀 Features

### ✔ Transfer Learning (MobileNetV2)

* Pretrained ImageNet weights
* Custom classification head
* Input size standardized to 224×224×3

### ✔ Explainable AI (XAI)

* Full **Grad-CAM implementation**
* Heatmap overlays on original MRI images
* Correct vs wrongly predicted comparisons

### ✔ Error Analysis

* Confusion matrix
* Misclassified image viewer
* Class distribution analysis
* Insights into model failure patterns

### ✔ Robust Data Pipeline

* Parquet-based dataset loading
* Custom grayscale image decoding
* Preprocessing validated with shape checks
* Safe transformer functions (`tf.image.grayscale_to_rgb`)

---

## 📂 Repository Structure

```
│── README.md
│── notebook.ipynb                  # (Your Kaggle / Colab notebook)
│── data/
│   ├── train.parquet
│   └── test.parquet
│── src/
│   ├── preprocessing.py             # Grayscale->RGB, resizing, TL prep
│   ├── mobilenet_model.py           # Model build & compile
│   ├── training.py                  # Training loop + callbacks
│   ├── evaluation.py                # ROC, F1, Confusion, metrics
│   └── gradcam.py                   # Explainability functions
│
└── outputs/
    ├── gradcam_correct/
    ├── gradcam_wrong/
    ├── confusion_matrix.png
    └── training_logs.json
```

---

## 📝 Installation

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn pillow pyarrow
```

This script runs the full project: training, testing, Grad-CAM, and error analysis.

Just run:
```bash
python run.py
```
If running in Kaggle/Colab, dependencies are mostly preinstalled.

---

## 📦 Dataset

The project uses MRI images stored inside `.parquet` files.

Each row contains:

* `image`: raw grayscale image bytes
* `label`: class name

Before training:

```python
train_df = pd.read_parquet("data/train.parquet")
test_df  = pd.read_parquet("data/test.parquet")
```

Images are decoded like this:

```python
def bytes_to_pixels(b):
    img = Image.open(io.BytesIO(b))
    return np.array(img)
```

---

## 🛠 Preprocessing Pipeline

### Step 1 — Ensure channel dimension

```python
def add_channel(img):
    if img.ndim == 2:
        return np.expand_dims(img, -1)
    return img
```

### Step 2 — Convert grayscale → RGB

```python
img = tf.image.grayscale_to_rgb(img)
```

### Step 3 — Resize to 224×224

### Step 4 — MobileNetV2 `preprocess_input`

Final preprocessed image shape:

```
(224, 224, 3)
```

---

## 🧪 Model (MobileNetV2 Transfer Learning)

```python
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(base.input, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

---

## 📈 Training

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)
```

---

## 🎯 Evaluation Metrics

* Accuracy
* F1-score
* Class-wise ROC curves
* Confusion matrix

Example confusion matrix:

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred)
```

---

## 🔥 Explainability (Grad-CAM)

The `gradcam.py` module contains:

### ✔ Automatic last conv layer detection

### ✔ Heatmap generation

### ✔ Overlay on original grayscale MRI

### ✔ Save utilities

Example:

```python
heatmap = make_gradcam_heatmap(img, model)
overlay = overlay_heatmap_on_grayscale(original_img, heatmap)
plt.imshow(overlay)
```

---

## 🧩 Error Analysis Summary

The notebook includes:

### ✔ Misclassified Image Viewer

Helpful for finding confusing MRI patterns.

### ✔ Class Imbalance Check

Shows if the model is biased.

### ✔ Feature Confusion Patterns

Grad-CAM reveals if the model focuses on:

* skull edges
* background noise
* wrong anatomical regions

---

## 💡 Why the Model Fails (Short Paragraph)

> The model sometimes misclassifies MRI images due to subtle intensity variations and overlapping features between classes. Certain classes are underrepresented, causing biased decision boundaries. Grad-CAM reveals that some wrong predictions occur because the model attends to irrelevant regions such as skull edges or background noise. Improved augmentation, balanced sampling, and fine-tuning deeper MobileNet layers significantly reduce these errors.

---

## 🏁 Results (Week 3 Deliverables)

✔ Grad-CAM heatmaps (saved in `/outputs/gradcam_*`)
✔ Explainability report
✔ Updated model with improved accuracy
✔ Error-analysis section with insights

---

## 🤝 Contributing

Pull requests, suggestions, and issue reports are welcome.

---

## 📜 License

MIT License

---


