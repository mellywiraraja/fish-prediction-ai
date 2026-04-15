import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Fish Counter AI", layout="centered")

MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1Y_K4r-TfEqnCEJ0uhrnTB9n7i299QyGS"
    gdown.download(url, MODEL_PATH, quiet=False)
    
# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "model.h5",
        compile=False,
        safe_mode=False
    )
    return model

model = load_model()

# =========================
# UI
# =========================
st.title("🐟 Fish Counter AI")
st.markdown("Prediksi jumlah benih ikan dari citra secara otomatis")

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    # =========================
    # PREPROCESSING (FIXED)
    # =========================
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32)   # ⬅️ TANPA /255
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # DEBUG (WAJIB UNTUK VALIDASI)
    # =========================
    st.write("Shape:", img_array.shape)
    st.write("Min-Max:", img_array.min(), img_array.max())

    # =========================
    # PREDICTION
    # =========================
    if st.button("Prediksi Jumlah Ikan"):
        prediction = model.predict(img_array)

        # Debug output model
        st.write("Raw Prediction:", prediction)

        # =========================
        # POSTPROCESSING
        # =========================
        fish_count = int(np.round(prediction[0][0]))

        st.success(f"Jumlah Ikan Terdeteksi: {fish_count}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with Streamlit & TensorFlow")