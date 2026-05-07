import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Anime Detector AI", 
    page_icon="🎨",
    layout="centered"
)

# --- 2. LOAD MODEL (CACHED) ---
@st.cache_resource
def load_trained_model():
    # Memastikan model dimuat sekali saja untuk menghemat memori
    return tf.keras.models.load_model('model_anime_v2.h5')

try:
    model = load_trained_model()
except Exception as e:
    st.error("⚠️ Model '.h5' tidak ditemukan di direktori! Pastikan file model sudah diunggah.")
    st.stop()

# --- 3. ANTARMUKA PENGGUNA (UI) ---
st.title("🎨 Anime Classifier: Manusia vs AI")
st.markdown("""
Deteksi keaslian gambar anime menggunakan arsitektur **MobileNetV2**. 
Sistem ini menganalisis pola guratan untuk membedakan hasil karya seniman dan generatif AI.
""")
st.divider()

# Pilihan Sumber Gambar: Responsif untuk Mobile
source_option = st.radio("Pilih Sumber Gambar:", ("Galeri (Upload)", "Kamera (Live)"))

input_image = None
if source_option == "Galeri (Upload)":
    input_image = st.file_uploader("Unggah gambar anime dari galeri...", type=["jpg", "jpeg", "png"])
else:
    input_image = st.camera_input("Ambil foto gambar anime langsung")

# --- 4. LOGIKA PEMROSESAN & KLASIFIKASI ---
if input_image is not None:
    # Tampilkan Gambar yang diinput
    image = Image.open(input_image)
    st.image(image, caption="Gambar yang akan dianalisis", use_container_width=True)
    
    with st.spinner('🔍 Sedang menganalisis pola mikroskopis...'):
        # A. Preprocessing (Standar MobileNetV2)
        img = image.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalisasi pixel
        img_array = np.expand_dims(img_array, axis=0)

        # B. Inferensi (Prediksi)
        prediction = model.predict(img_array)
        score = prediction[0][0]

        # C. Logika Ambang Batas Keyakinan (Confidence Threshold)
        # Menghitung seberapa yakin model (0.0 - 1.0)
        certainty = abs(score - 0.5) * 2 

        st.divider()

        # D. Filter "Bukan Anime" / Ketidakpastian
        if certainty < 0.25: # Jika model sangat ragu (skor dekat 0.5)
            st.warning("⚠️ **Hasil Tidak Pasti.**")
            st.write("Model ragu apakah ini anime atau objek lain. Pastikan gambar yang diunggah adalah ilustrasi anime yang jelas.")
            st.info(f"Tingkat Keyakinan Rendah: {certainty * 100:.2f}%")
        
        else:
            # E. Tampilkan Hasil Klasifikasi
            if score > 0.5:
                st.success(f"### HASIL: **KARYA MANUSIA**")
                st.write("Model mendeteksi karakteristik guratan tangan manusia.")
                st.progress(score) # Bar visualitas
                st.info(f"Tingkat Keyakinan: {score * 100:.2f}%")
            else:
                st.error(f"### HASIL: **TERDETEKSI AI**")
                st.write("Model mendeteksi pola repetitif khas generatif AI.")
                st.progress(1 - score)
                st.info(f"Tingkat Keyakinan: {(1 - score) * 100:.2f}%")

# --- 5. FOOTER ---
st.divider()
st.caption("🚀 Powered by MobileNetV2 Architecture | Optimized for Mobile Devices")
