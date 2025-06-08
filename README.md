# 🇮🇩 Sentiment Analysis Project – Bahasa Indonesia

Analisis sentimen teks berbahasa Indonesia menggunakan metode Support Vector Machine (SVM) dan model deep learning Bidirectional Long Short-Term Memory (BiLSTM) berbasis embedding FastText. Proyek ini mencakup tahap persiapan data, pelatihan model, evaluasi, serta prediksi sentimen teks secara supervised learning.

---

## 📖 Abstrak

Analisis sentimen merupakan cabang penting dalam pengolahan bahasa alami (Natural Language Processing/NLP) yang bertujuan mengklasifikasi opini atau perasaan pengguna terhadap suatu topik. Pada proyek ini, digunakan dua pendekatan utama: model SVM sebagai metode machine learning klasik dan BiLSTM sebagai model deep learning yang mampu menangkap konteks urutan kata secara bidirectional. Data teks berbahasa Indonesia diproses dan direpresentasikan dengan FastText embeddings untuk meningkatkan kualitas representasi fitur.

---

## 🗂 Struktur Direktori

sentiment-analysis-project/
├── data/              # Dataset ulasan (tidak disertakan di GitHub)
├── figures/           # Visualisasi hasil evaluasi model
├── models/            # Model terlatih (.h5 dan .pkl)
├── nltk_data/         # Data tambahan NLTK untuk preprocessing
├── notebooks/         # Notebook eksplorasi dan pelatihan
├── reports/           # Laporan hasil analisis dan evaluasi model
├── src/               # Skrip Python utama (training, prediction, utils)
├── main.py            # Skrip utama untuk pelatihan dan evaluasi
├── streamlit_app.py   # Aplikasi Streamlit untuk demo interaktif
├── requirements.txt   # Daftar dependensi Python
├── runtime.txt        # Konfigurasi runtime (misalnya untuk deployment)
├── .gitignore         # File/folder yang diabaikan Git
└── README.md          # Dokumentasi proyek


## ⚠️ Catatan Penting: Data Tidak Diunggah ke GitHub

Folder `data/` **tidak disertakan** dalam repositori ini karena ukuran dataset yang besar dan kebijakan GitHub mengenai batas ukuran file.

Silakan unduh dataset secara manual melalui tautan berikut dan letakkan pada folder lokal `data/`:

| Folder  | Isi Dataset                      | Link Unduh                                                                                       |
| ------- | -------------------------------- | ------------------------------------------------------------------------------------------------ |
| data/   | Dataset pelatihan dan pengujian   | [🔗 Download Dataset](https://drive.google.com/drive/folders/13cAl19bYk4aedJhtWVPkAHVypzTBuBGN?usp=sharing)

---

## 🧪 Metode yang Digunakan

- **Support Vector Machine (SVM):**  
  Metode klasifikasi teks berbasis fitur TF-IDF dan embedding yang efektif untuk klasifikasi sentimen.

- **Bidirectional Long Short-Term Memory (BiLSTM):**  
  Model deep learning dengan kemampuan memproses konteks kata dari dua arah, sehingga dapat memahami nuansa dalam kalimat lebih baik.

- **FastText Embeddings:**  
  Representasi kata yang memperhitungkan sub-kata sehingga cocok untuk bahasa Indonesia yang kaya akan morfologi.

---


## 🚀 Cara Menjalankan Proyek

1. Clone repositori dan masuk ke folder proyek:
bash
git clone https://github.com/namamu/sentiment-analysis-project.git
cd sentiment-analysis-project

2. Siapkan virtual environment dan aktifkan
python -m venv venv
Untuk Linux/macOS
source venv/bin/activate
Untuk Windows
venv\Scripts\activate

3. Instal semua dependensi:
bash
Copy
Edit
pip install -r requirements.txt

4. Unduh dataset secara manual dan tempatkan pada folder data/.

5. Jalankan skrip utama untuk training, evaluasi, atau prediksi:
bash
Copy
Edit
python main.py

6. Atau jalankan aplikasi demo interaktif menggunakan Streamlit:
bash
Copy
Edit
streamlit run streamlit_app.py

---

## 📊 Evaluasi dan Visualisasi
Visualisasi hasil evaluasi model tersedia di folder figures/, termasuk grafik akurasi, confusion matrix, dan metrik lainnya untuk membantu memahami performa model.

---


## 📚 Teknologi yang Digunakan
Python 3.12+
FastText word embeddings (cc.id.300.vec)
TensorFlow / Keras (untuk BiLSTM)
Scikit-learn (untuk SVM dan evaluasi)
NumPy & Pandas (manipulasi data)
Matplotlib & Seaborn (visualisasi)
Streamlit (untuk aplikasi interaktif)

---

## 🤝 Kontribusi
Kami sangat mengapresiasi kontribusi dari komunitas. Silakan fork repositori ini, buat branch baru, dan ajukan pull request dengan deskripsi yang jelas.

---

## 📄 Lisensi
Proyek ini dilisensikan di bawah MIT License. Silakan lihat file LICENSE untuk detail lebih lanjut.

