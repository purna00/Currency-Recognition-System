import os
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_ROOT = Path("Indian currency dataset v1")

TRAIN_DIR = DATASET_ROOT / "training"
VAL_DIR = DATASET_ROOT / "validation"

MODEL_DIR = Path("model")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 8
FINE_TUNE_LAYERS = 20

SEED = 42

SELECTED_CLASSES = [
    "10",
    "20",
    "50",
    "100",
    "200",
    "500"
]

MODEL_DIR.mkdir(exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 2. CHECK DATASET PATHS
# ============================================================

print("\nChecking dataset...\n")

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training folder not found:\n{TRAIN_DIR}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation folder not found:\n{VAL_DIR}"
    )


# ============================================================
# 3. FIND IMAGE FILES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_images(folder, class_names):

    image_paths = []
    labels = []

    for label, class_name in enumerate(class_names):

        class_folder = folder / class_name

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Class folder not found:\n{class_folder}"
            )

        files = sorted(
            [
                file
                for file in class_folder.iterdir()
                if file.is_file()
                and file.suffix.lower()
                in IMAGE_EXTENSIONS
            ]
        )

        print(
            f"{class_name}: "
            f"{len(files)} images"
        )

        for file in files:
            image_paths.append(str(file))
            labels.append(label)

    return image_paths, labels


# ============================================================
# 4. LOAD TRAINING AND VALIDATION FILES
# ============================================================

print("Training dataset:")

train_paths, train_labels = get_images(
    TRAIN_DIR,
    SELECTED_CLASSES
)

print("\nValidation dataset:")

val_paths, val_labels = get_images(
    VAL_DIR,
    SELECTED_CLASSES
)

print("\n======================================")
print("DATASET SUMMARY")
print("======================================")

print(
    f"Total training images: "
    f"{len(train_paths)}"
)

print(
    f"Total validation images: "
    f"{len(val_paths)}"
)

print(
    f"Number of classes: "
    f"{len(SELECTED_CLASSES)}"
)

print("======================================\n")


# ============================================================
# 5. CLASS WEIGHTS
# ============================================================

train_counts = {}

for class_index, class_name in enumerate(
    SELECTED_CLASSES
):

    train_counts[class_name] = sum(
        1
        for label in train_labels
        if label == class_index
    )


total_train = len(train_labels)
num_classes = len(SELECTED_CLASSES)

class_weight = {}

for class_index, class_name in enumerate(
    SELECTED_CLASSES
):

    count = train_counts[class_name]

    class_weight[class_index] = (
        total_train /
        (num_classes * count)
    )


print("Class weights:")

for index, weight in class_weight.items():

    print(
        f"Rs.{SELECTED_CLASSES[index]}: "
        f"{weight:.3f}"
    )


# ============================================================
# 6. IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ============================================================
# 7. CREATE TF.DATA DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)


train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


train_dataset = train_dataset.shuffle(
    buffer_size=min(
        len(train_paths),
        1000
    ),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)


train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# 8. DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)


# ============================================================
# 9. LOAD MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...\n")

base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ============================================================
# 10. BUILD MODEL
# ============================================================

inputs = keras.Input(
    shape=IMG_SIZE + (3,),
    name="currency_image"
)

x = data_augmentation(inputs)

x = keras.applications.mobilenet_v2.preprocess_input(
    x
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax",
    name="currency_prediction"
)(x)

model = keras.Model(
    inputs,
    outputs,
    name="currency_recognition_model"
)


# ============================================================
# 11. COMPILE INITIAL MODEL
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 12. MODEL SUMMARY
# ============================================================

print("\nModel summary:\n")

model.summary()


# ============================================================
# 13. CALLBACKS
# ============================================================

best_model_path = (
    MODEL_DIR /
    "best_currency_model.keras"
)


def create_callbacks():

    return [

        keras.callbacks.ModelCheckpoint(
            filepath=str(
                best_model_path
            ),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]


# ============================================================
# 14. INITIAL TRAINING
# ============================================================

print("\n======================================")
print("STARTING INITIAL TRAINING")
print("======================================\n")

history_initial = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weight,
    callbacks=create_callbacks()
)


# ============================================================
# 15. FINE-TUNING
# ============================================================

print("\n======================================")
print("STARTING FINE-TUNING")
print("======================================\n")

base_model.trainable = True


# Freeze all layers except the last few
for layer in base_model.layers[
    :-FINE_TUNE_LAYERS
]:
    layer.trainable = False


# Keep BatchNormalization layers frozen
for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False


# Recompile with a smaller learning rate
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weight,
    callbacks=create_callbacks()
)


# ============================================================
# 16. LOAD BEST MODEL
# ============================================================

print("\nLoading best model...\n")

best_model = keras.models.load_model(
    str(best_model_path)
)


# ============================================================
# 17. FINAL VALIDATION
# ============================================================

print("\nEvaluating final model...\n")

loss, accuracy = best_model.evaluate(
    val_dataset,
    verbose=1
)


# ============================================================
# 18. SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR /
    "currency_model.keras"
)

best_model.save(
    str(final_model_path)
)


# ============================================================
# 19. SAVE CLASS NAMES
# ============================================================

class_names_path = (
    MODEL_DIR /
    "class_names.json"
)

with open(
    class_names_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        SELECTED_CLASSES,
        file,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 20. FINAL RESULTS
# ============================================================

print("\n======================================")
print("TRAINING COMPLETED")
print("======================================")

print(
    f"\nValidation Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Validation Loss: "
    f"{loss:.4f}"
)

print(
    "\nBest model:"
)

print(
    best_model_path
)

print(
    "\nFinal model:"
)

print(
    final_model_path
)

print(
    "\nClasses:"
)

for index, name in enumerate(
    SELECTED_CLASSES
):

    print(
        f"{index}: Rs.{name}"
    )

print(
    "\nTraining finished successfully!"
)