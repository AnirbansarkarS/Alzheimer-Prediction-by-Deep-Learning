import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import seaborn as sns
import os


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()


def evaluate_model(model, X_val, y_val, class_names):
    y_proba = model.predict(X_val)
    y_pred = np.argmax(y_proba, axis=1)

    print(classification_report(y_val, y_pred, target_names=class_names))

    plot_confusion_matrix(y_val, y_pred, class_names)

    # ROC AUC (macro)
    try:
        roc_score = roc_auc_score(
            y_val,
            y_proba,
            multi_class="ovo",
            average="macro"
        )
        print("ROC-AUC Score:", roc_score)
    except:
        print("ROC-AUC could not be computed (class count issue).")

    return y_pred, y_proba
