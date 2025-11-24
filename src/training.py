import json
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def train_model(model, X_train, y_train, X_val, y_val, save_path="outputs/best_model.h5"):
    """Training loop + callbacks"""

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
        ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=callbacks
    )

    # Save training logs
    with open("outputs/training_logs.json", "w") as f:
        json.dump(history.history, f)

    return model, history
