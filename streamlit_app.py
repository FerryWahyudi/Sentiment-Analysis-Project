import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nlp_id.lemmatizer import Lemmatizer
from indoNLP.preprocessing import replace_slang, replace_word_elongation

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Load models and preprocessors ===
svm_model = joblib.load("models/svm_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

bilstm_model = load_model("models/bilstm_model.h5")
tokenizer = joblib.load("models/tokenizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

lemmatizer = Lemmatizer()
stop_words = set(stopwords.words('indonesian'))


# === Preprocessing ===
def clean_input(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = replace_slang(text)
    text = replace_word_elongation(text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)

# === Utility Functions ===
def get_top_keywords(text, tfidf, top_n=5):
    feature_names = tfidf.get_feature_names_out()
    tfidf_vector = tfidf.transform([text])
    sorted_items = np.argsort(tfidf_vector.toarray()[0])[::-1]
    top_keywords = [feature_names[i] for i in sorted_items[:top_n] if tfidf_vector[0, i] > 0]
    return top_keywords

def predict_svm(text):
    X = tfidf.transform([text])
    pred = svm_model.predict(X)[0]
    prob = svm_model.predict_proba(X)[0]
    top_words = get_top_keywords(text, tfidf)
    return pred, prob, top_words

def predict_bilstm(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    prob = bilstm_model.predict(padded, verbose=0)[0]
    label = label_encoder.inverse_transform([np.argmax(prob)])[0]
    return label, prob

# === Streamlit UI ===
st.set_page_config(page_title="Sentiment Analysis - Gojek", page_icon="🛵")

st.title("Sentiment Analysis for Gojek App Review")
st.markdown("Masukkan ulasan pengguna Gojek dan pilih model untuk memprediksi sentimennya (positif, netral, atau negatif).")

# === Main Input UI ===
examples = {
    "Pesan cepat, driver ramah, aplikasi sangat membantu": "positif",
    "Driver datang tepat waktu dan pelayanan bagus": "positif",
    "Harga terlalu mahal untuk jarak dekat": "negatif",
    "Aplikasi sering error saat pemesanan": "negatif",
    "Lumayan, tapi kadang lambat": "netral"
}

example_options = [""] + list(examples.keys())
selected_example = st.selectbox("###### 💬 Pilih salah satu contoh ulasan", options=example_options)

st.markdown("###### 📝 Atau tulis ulasanmu sendiri")
input_text = st.text_area("Tulis ulasan pengguna di bawah ini", value=selected_example if selected_example else "", height=150)

# === Sidebar ===
st.sidebar.header("⚙️ Pengaturan Model")
model_choice = st.sidebar.radio(
    "Pilih Model Analisis Sentimen",
    options=["SVM", "BiLSTM"],
    format_func=lambda x: "SVM (TF-IDF) - ML Tradisional" if x == "SVM" else "BiLSTM - Deep Learning"
)
compare_models = st.sidebar.checkbox("Bandingkan Hasil Dua Model")

# === Prediksi ===
if st.button("Prediksi Sentimen"):
    if not input_text.strip():
        st.warning("⚠️ Silakan masukkan ulasan terlebih dahulu.")
    else:
        cleaned = clean_input(input_text)

        if compare_models:
            st.markdown("## 🔄 Komparasi Kedua Model")
            col1, col2 = st.columns(2)

            # === SVM ===
            with col1:
                try:
                    pred_svm, prob_svm, keywords = predict_svm(cleaned)
                    color_svm = "#2ecc71" if pred_svm.lower() == "positive" else "#e74c3c" if pred_svm.lower() == "negative" else "#f1c40f"

                    st.markdown(
                        f"<p style='font-size:16px;'>💡 <b>SVM</b>: Sentimen = "
                        f"<span style='color:{color_svm}; font-weight:bold;'>{pred_svm.capitalize()}</span></p>",
                        unsafe_allow_html=True
                    )

                    prob_dict_svm = dict(zip(svm_model.classes_, prob_svm))
                    df_svm = pd.DataFrame(prob_dict_svm.items(), columns=["Sentimen", "Probabilitas"]).sort_values(by="Probabilitas", ascending=False)
                    st.bar_chart(df_svm.set_index("Sentimen"))

                    st.markdown("###### 📋 Confidence Score (SVM):")
                    for label, p in prob_dict_svm.items():
                        st.markdown(f"- {label.capitalize()}: **{p * 100:.2f}%**")

                    if keywords:
                        st.markdown("###### 🔹 Kata Kunci (TF-IDF):")
                        st.markdown(" → " + ", ".join(f"`{kw}`" for kw in keywords))

                except Exception as e:
                    st.error(f"❌ Error prediksi SVM: {e}")

            # === BiLSTM ===
            with col2:
                try:
                    label_bilstm, prob_bilstm = predict_bilstm(cleaned)
                    color_bilstm = "#2ecc71" if label_bilstm.lower() == "positive" else "#e74c3c" if label_bilstm.lower() == "negative" else "#f1c40f"

                    st.markdown(
                        f"<p style='font-size:16px;'>🧠 <b>BiLSTM</b>: Sentimen = "
                        f"<span style='color:{color_bilstm}; font-weight:bold;'>{label_bilstm.capitalize()}</span></p>",
                        unsafe_allow_html=True
                    )

                    prob_dict_bilstm = dict(zip(label_encoder.classes_, prob_bilstm))
                    df_bilstm = pd.DataFrame(prob_dict_bilstm.items(), columns=["Sentimen", "Probabilitas"]).sort_values(by="Probabilitas", ascending=False)
                    st.bar_chart(df_bilstm.set_index("Sentimen"))

                    st.markdown("###### 📋 Confidence Score (BiLSTM):")
                    for label, p in prob_dict_bilstm.items():
                        st.markdown(f"- {label.capitalize()}: **{p * 100:.2f}%**")

                except Exception as e:
                    st.error(f"❌ Error prediksi BiLSTM: {e}")

            try:
                if pred_svm.lower() != label_bilstm.lower():
                    st.warning("⚠️ Kedua model memprediksi sentimen yang berbeda.")
                else:
                    st.success("✅ Kedua model memberikan prediksi yang sama.")
            except:
                pass

        else:
            # MODE INDIVIDU
            if model_choice == "SVM":
                try:
                    prediction, prob, keywords = predict_svm(cleaned)
                    color = "#2ecc71" if prediction.lower() == "positive" else "#e74c3c" if prediction.lower() == "negative" else "#f1c40f"

                    st.markdown(f"<p style='font-size:18px;'>💬 Hasil Prediksi (SVM): Sentimen = <span style='color:{color}; font-weight:bold;'>{prediction.capitalize()}</span></p>", unsafe_allow_html=True)

                    prob_dict = dict(zip(svm_model.classes_, prob))
                    prob_df = pd.DataFrame(prob_dict.items(), columns=["Sentimen", "Probabilitas"]).sort_values(by="Probabilitas", ascending=False)
                    st.bar_chart(prob_df.set_index("Sentimen"))

                    st.markdown("##### 📋 Nilai Confidence Score:")
                    for label, p in prob_dict.items():
                        st.markdown(f"- {label.capitalize()}: **{p * 100:.2f}%**")

                    if keywords:
                        st.markdown("##### 🔹 Kata Kunci Utama (TF-IDF):", unsafe_allow_html=True)
                        st.markdown(" → " + ", ".join(f"`{kw}`" for kw in keywords))

                except Exception as e:
                    st.error(f"❌ Terjadi error saat prediksi SVM: {e}")

            else:
                try:
                    label, prob = predict_bilstm(cleaned)
                    color = "#2ecc71" if label.lower() == "positive" else "#e74c3c" if label.lower() == "negative" else "#f1c40f"

                    st.markdown(f"<p style='font-size:18px;'>💬 Hasil Prediksi (BiLSTM): Sentimen = <span style='color:{color}; font-weight:bold;'>{label.capitalize()}</span></p>", unsafe_allow_html=True)

                    prob_dict = dict(zip(label_encoder.classes_, prob))
                    prob_df = pd.DataFrame(prob_dict.items(), columns=["Sentimen", "Probabilitas"]).sort_values(by="Probabilitas", ascending=False)
                    st.bar_chart(prob_df.set_index("Sentimen"))

                    st.markdown("##### 📋 Nilai Confidence Score:")
                    for label, p in prob_dict.items():
                        st.markdown(f"- {label.capitalize()}: **{p * 100:.2f}%**")

                except Exception as e:
                    st.error(f"❌ Terjadi error saat prediksi BiLSTM: {e}")
