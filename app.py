import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the trained model (cached so it won’t reload each time)
@st.cache_resource
def load_digit_model():
    return load_model("digit_model.h5")

model = load_digit_model()

st.title("✍️ MNIST Digit Classifier")
st.write("Upload a digit image **or** draw a digit below to classify it (0–9).")

# -----------------------
# File uploader section
# -----------------------
uploaded_file = st.file_uploader("Upload a 28x28 digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image = image.resize((28, 28))  # resize
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.image(image, caption=f"Predicted Digit: {predicted_label} ({confidence:.2f}% confidence)", width=150)
    st.bar_chart(prediction[0])

st.markdown("---")

# -----------------------
# Drawing canvas section
# -----------------------
st.subheader("Or draw a digit below:")

canvas_result = st_canvas(
    fill_color="black",      # background
    stroke_width=15,         # thickness of stroke
    stroke_color="white",    # digit color
    background_color="black",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Convert canvas to grayscale 28x28
    img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype("uint8"))  # take RGB
    img = img.convert("L")  # grayscale
    img = img.resize((28, 28))  # resize to MNIST size
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.image(img, caption="Processed Digit", width=150)
    st.success(f"Predicted Digit: **{predicted_label}** ({confidence:.2f}% confidence)")
    st.bar_chart(prediction[0])
