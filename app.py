import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Anime Detector AI", layout="centered", page_icon="🎨")

# --- 2. LOAD MODEL (CACHED) ---
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('model_anime_v2_asli.h5')

try:
    model = load_trained_model()
except Exception as e:
    st.error("Model 'model_anime_v2_asli.h5' tidak ditemukan!")
    st.stop()

# --- 3. TAMPILAN ANTARMUKA ---
st.title(" Anime Classifier : Multi-Upload Mode")
st.markdown("Unggah beberapa gambar sekaligus untuk deteksi massal.")
st.divider()

# Parameter Utama: accept_multiple_files=True
uploaded_files = st.file_uploader(
    "Pilih satu atau lebih gambar anime...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# --- 4. PROSES KLASIFIKASI BERUNTUN ---
if uploaded_files:
    st.write(f" Menghitung total: **{len(uploaded_files)} gambar**")
    
    # Membuat container untuk hasil agar rapi
    for uploaded_file in uploaded_files:
        # Menggunakan expander agar hasil tiap gambar bisa dibuka-tutup (scannable)
        with st.expander(f"📷 Hasil Analisis: {uploaded_file.name}"):
            
            # 1. Load & Tampilkan Gambar (Ukuran kecil di expander)
            image = Image.open(uploaded_file)
            st.image(image, width=250)
            
            # 2. Preprocessing
            img = image.convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # 3. Inferensi
            prediction = model.predict(img_array)
            score = float(np.clip(prediction[0][0], 0.0, 1.0))
            certainty = abs(score - 0.5) * 2

            # 4. Filter Eliminasi (Threshold 0.4)
            if certainty < 0.4:
                st.warning("❌ **Ditolak: Bukan Karakter Anime**")
                st.caption(f"Keyakinan model terlalu rendah ({certainty*100:.2f}%)")
            else:
                if score > 0.5:
                    st.success(f"✅ **HASIL: KARYA MANUSIA**")
                    st.progress(score)
                    st.write(f"Akurasi Prediksi: {score * 100:.2f}%")
                else:
                    st.error(f"🤖 **HASIL: TERDETEKSI AI**")
                    val_ai = 1.0 - score
                    st.progress(val_ai)
                    st.write(f"Akurasi Prediksi: {val_ai * 100:.2f}%")

st.divider()
st.caption("Batch Processing Mode | Powered by MobileNetV2")
