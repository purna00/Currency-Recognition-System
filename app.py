import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Currency Recognition System",
    page_icon="💰",
    layout="centered"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/best_currency_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"

IMG_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_and_classes():

    model = keras.models.load_model(
        MODEL_PATH
    )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(file)

    return model, class_names


model, class_names = load_model_and_classes()


# ============================================================
# TITLE
# ============================================================

st.title("💰 Currency Recognition Project")

st.write(
    "Upload an Indian currency note image "
    "and the AI model will identify its denomination."
)


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a currency image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # Display uploaded image
    st.subheader("Uploaded Image")

    st.image(
        uploaded_file,
        caption="Currency Image",
        use_container_width=True
    )

    # Read image
    image = tf.keras.utils.load_img(
        uploaded_file,
        target_size=IMG_SIZE
    )

    # Convert to array
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
        .preprocess_input(
            image_array
        )
    )

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_currency = (
        class_names[predicted_index]
    )

    confidence = (
        float(predictions[predicted_index])
        * 100
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("Recognition Result")

    st.success(
        f"Predicted Currency: ₹{predicted_currency}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    # ========================================================
    # ALL PREDICTIONS
    # ========================================================

    st.subheader(
        "Prediction Probabilities"
    )

    for index, probability in enumerate(
        predictions
    ):

        st.write(
            f"₹{class_names[index]}"
        )

        st.progress(
            float(probability)
        )

        st.caption(
            f"{probability * 100:.2f}%"
        )


# ============================================================
# INFORMATION
# ============================================================

else:

    st.info(
        "Please upload a currency image to begin."
    )