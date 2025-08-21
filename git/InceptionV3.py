import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# === Konfigurasi Model ===
MODEL_PATH = "fine_tuned_model_fold_5.keras"
CLASS_LABELS = ['Esophagitis', 'normal', 'polips', 'ulcerative_colitis']
IMAGE_SIZE = (299, 299)  

def load_trained_model(path):
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.sidebar.error(f" Gagal memuat model: {str(e)}")
        st.stop()

def preprocess_image(img, target_size):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def predict_image(model, img_array, class_labels):
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = class_labels[class_index]
    probability = np.max(prediction) * 100
    return class_label, probability, prediction[0]

def plot_prediction_bar(probabilities, class_labels):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(class_labels, probabilities * 100, color=['darkred', 'green', 'orange', 'red'])
    ax.set_xlabel("Probabilitas (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Probabilitas Model untuk Tiap Kelas")
    plt.tight_layout()
    st.pyplot(fig)

def add_to_history(history, label, probability):
    history.append({"label": label, "prob": probability})
    st.session_state['history'] = history

def generate_prediction_pdf(label, prob, class_labels, probabilities):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 2 * cm, "Laporan Hasil Prediksi Penyakit Usus")


    c.setFont("Helvetica", 12)
    text = c.beginText(2.5 * cm, height - 3.5 * cm)
    text.textLine(f"Hasil Prediksi: {label}")
    text.textLine(f"Tingkat Keyakinan: {prob:.2f}%")
    text.textLine("")
    text.textLine("Probabilitas untuk Setiap Kelas:")

    for cls, p in zip(class_labels, probabilities):
        text.textLine(f"- {cls}: {p*100:.2f}%")

    text.textLine("")
    text.textLine("Catatan: Ini adalah prediksi dari model, bukan diagnosis medis.")
    c.drawText(text)

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawCentredString(width / 2, 1.5 * cm, "Aplikasi Deteksi Penyakit Usus — 2025")

    c.showPage()
    c.save()

    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

st.set_page_config(page_title="Deteksi Penyakit Usus", layout="centered")
st.title("🔬 Deteksi Penyakit Usus dari Gambar")
st.write("Unggah gambar endoskopi untuk mendeteksi apakah sehat atau memiliki penyakit.")


if 'history' not in st.session_state:
    st.session_state['history'] = []


# === Sidebar ===
with st.sidebar:
    st.header("ℹ️` Informasi Model")
    st.write(f"Model: `{MODEL_PATH}`")
    st.write("Format gambar yang didukung: **JPG, PNG, JPEG**")
    st.warning("⚠️ Model ini hanya alat bantu dan **bukan diagnosis medis**.")

    st.divider()
    st.subheader("🕑 Riwayat Prediksi")
    if st.session_state['history']:
        for i, record in enumerate(reversed(st.session_state['history']), 1):
            st.write(f"{i}. **{record['label']}** ({record['prob']:.2f}%)")
    else:
        st.info("Belum ada riwayat prediksi.")



model = load_trained_model(MODEL_PATH)

uploaded_file = st.file_uploader(" Unggah Gambar Endoskopi", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Gambar yang Diunggah", width=400)

        img_array = preprocess_image(img, IMAGE_SIZE)

        with st.spinner("🔍 Melakukan prediksi..."):
            class_label, probability, probabilities = predict_image(model, img_array, CLASS_LABELS)

        st.success(f"**Hasil Prediksi:** {class_label}")
        st.metric(label="Tingkat Keyakinan", value=f"{probability:.2f}%")


        add_to_history(st.session_state['history'], class_label, probability)


        plot_prediction_bar(probabilities, CLASS_LABELS)


        pred_pdf = generate_prediction_pdf(class_label, probability, CLASS_LABELS, probabilities)

        st.download_button(
            label="📄 Download Laporan Prediksi ",
            data=pred_pdf,
            file_name='hasil_prediksi.pdf',
            mime='application/pdf'
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan dalam prediksi: {str(e)}")

st.markdown("---")
st.caption("© 2025. Aplikasi ini dikembangkan untuk keperluan tugas akhir klasifikasi penyakit usus.")
