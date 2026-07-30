import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/best_currency_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"

IMG_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading currency recognition model...")

model = keras.models.load_model(MODEL_PATH)

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:
    class_names = json.load(file)

print("Model loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_currency(image_path):

    if not os.path.isfile(image_path):

        print(
            f"\nImage not found:\n{image_path}"
        )

        return

    # Load image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    # Convert image to NumPy array
    image_array = tf.keras.utils.img_to_array(
        image
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    image_array = (
        keras.applications.mobilenet_v2
        .preprocess_input(image_array)
    )

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Find highest probability
    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = (
        float(predictions[predicted_index])
        * 100
    )

    predicted_currency = (
        class_names[predicted_index]
    )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n======================================")
    print("     CURRENCY RECOGNITION RESULT")
    print("======================================")

    print(
        f"\nPredicted Currency: "
        f"Rs.{predicted_currency}"
    )

    print(
        f"Confidence: "
        f"{confidence:.2f}%"
    )

    print("\nAll Predictions:")

    for index, probability in enumerate(
        predictions
    ):

        print(
            f"Rs.{class_names[index]}: "
            f"{probability * 100:.2f}%"
        )

    print("\n======================================")


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    image_path = input(
        "\nEnter the path of the currency image: "
    ).strip().strip('"')

    predict_currency(image_path)