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
st.title("🎨 Anime Klasifikasi")
st.markdown("Unggah dari galeri atau ambil foto langsung menggunakan kamera.")
st.divider()

# --- FITUR BARU: KAMERA (Optimasi Mobile) ---
camera_photo = st.camera_input("📸 Ambil foto gambar anime (Live)")

# Fitur Multi-Upload (Tetap Ada)
uploaded_files = st.file_uploader(
    "Pilih satu atau lebih gambar anime dari galeri...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

# --- 4. PROSES KLASIFIKASI BERUNTUN ---
# Gabungkan input kamera dan upload ke dalam satu list untuk diproses
all_images = []

if camera_photo:
    all_images.append(camera_photo)

if uploaded_files:
    all_images.extend(uploaded_files)

if all_images:
    st.write(f"📊 Menghitung total: **{len(all_images)} gambar**")
    
    for uploaded_file in all_images:
        # Penamaan judul expander
        label_name = getattr(uploaded_file, 'name', 'Hasil Foto Kamera')
        
        with st.expander(f"📷 Analisis: {label_name}"):
            
            # 1. Load & Tampilkan Gambar
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
                st.warning("⚠️ **Ditolak: Bukan Karakter Anime**")
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
st.caption("Batch Processing Mode | Mobile Optimized | Powered by MobileNetV2")
