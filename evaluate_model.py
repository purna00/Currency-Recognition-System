import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

VAL_DIR = Path("Indian currency dataset v1") / "validation"
MODEL_PATH = "model/best_currency_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = keras.models.load_model(MODEL_PATH)

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:
    class_names = json.load(file)

print("Model loaded successfully.")


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# COLLECT TRUE LABELS
# ============================================================

y_true = []

for _, labels in val_ds:
    y_true.extend(
        labels.numpy().tolist()
    )

y_true = np.array(y_true)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    val_ds,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n======================================")
print("MODEL EVALUATION")
print("======================================")

print(
    f"\nValidation Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            f"Rs.{name}"
            for name in class_names
        ],
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:\n")
print(cm)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Currency Recognition Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    [f"Rs.{x}" for x in class_names],
    rotation=45
)

plt.yticks(
    tick_marks,
    [f"Rs.{x}" for x in class_names]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


# Write values inside cells
for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center"
        )


plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

print(
    "\nConfusion matrix saved as:"
    "\nconfusion_matrix.png"
)