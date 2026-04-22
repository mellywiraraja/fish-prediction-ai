import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Fish Counter AI", layout="centered")

MODEL_PATH = "model.h5"

# =========================
# DOWNLOAD MODEL
# =========================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model..."):
        url = "https://drive.google.com/uc?id=1Y_K4r-TfEqnCEJ0uhrnTB9n7i299QyGS"
        gdown.download(url, MODEL_PATH, quiet=False)

# =========================
# LOAD MODEL (FIX BATCH_SHAPE)
# =========================
@st.cache_resource
def load_model():
    try:
        import h5py
        from tensorflow.keras.models import model_from_json

        # Buka file h5
        with h5py.File(MODEL_PATH, 'r') as f:
            model_config = f.attrs.get('model_config')
            if model_config is None:
                raise ValueError("Model config tidak ditemukan")

            model_config = model_config.decode('utf-8')

        # HAPUS batch_shape manual
        model_config = model_config.replace('"batch_shape": [null, 224, 224, 3],', '')
        model_config = model_config.replace('"batch_input_shape": [null, 224, 224, 3],', '')

        # Rebuild model
        model = model_from_json(model_config)
        model.load_weights(MODEL_PATH)

        return model

    except Exception as e:
        return str(e)

model = load_model()

# =========================
# UI
# =========================
st.title("🐟 Fish Counter AI")
st.markdown("Prediksi jumlah benih ikan dari citra secara otomatis")

# =========================
# HANDLE ERROR
# =========================
if isinstance(model, str):
    st.error("❌ Model gagal dimuat")
    st.code(model)
    st.stop()

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Hitung Jumlah Ikan"):
        with st.spinner("Memproses..."):
            prediction = model.predict(img_array)
            fish_count = max(0, int(np.round(prediction[0][0])))

        st.success(f"Jumlah ikan: {fish_count}")

st.markdown("---")
st.caption("Built with Streamlit & TensorFlow")
