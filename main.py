import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import joblib

from src.feature_engineering import get_tfidf_features, get_bilstm_features, load_fasttext_embeddings
from src.svm_model import train_svm, evaluate_svm, cross_validate_svm
from src.bilstm_model import build_bilstm_model, train_bilstm_model, evaluate_bilstm
from src.utils import save_model_pickle, save_model_h5

def main():
    print("\n📦 Sentiment Analysis Pipeline Started")

    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Load data
    print("\n📥 Loading data...")
    df = pd.read_csv("data/processed/Cleaned_GojekAppReview1_baru.csv")
    df = df.dropna(subset=["content_processed", "sentiment"])

    X_text = df["content_processed"]
    y = df["sentiment"]

    # === TF-IDF + SVM ===
    print("\n🔍 Step 1: TF-IDF + SVM")
    X_tfidf, tfidf_vectorizer = get_tfidf_features(X_text, return_vectorizer=True)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42)

    svm_model = train_svm(X_train_svm, y_train_svm)
    evaluate_svm(svm_model, X_test_svm, y_test_svm, save_dir="figures")
    cross_validate_svm(svm_model, X_tfidf, y, cv=5, save_dir="figures")

    save_model_pickle(svm_model, "models/svm_model.pkl")
    save_model_pickle(tfidf_vectorizer, "models/tfidf_vectorizer.pkl")

    # === Tokenizer + BiLSTM (with FastText) ===
    print("\n🧠 Step 2: Tokenizer + BiLSTM (FastText)")
    X_pad, y_onehot, tokenizer, label_encoder = get_bilstm_features(X_text, y)
    X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
        X_pad, y_onehot, test_size=0.2, random_state=42)

    embedding_matrix = load_fasttext_embeddings(
        filepath="data/word_embeddings/cc.id.300.vec",
        tokenizer=tokenizer,
        embedding_dim=300
    )

    bilstm_model = build_bilstm_model(
        vocab_size=len(tokenizer.word_index) + 1,
        embedding_dim=300,
        input_length=100,
        embedding_matrix=embedding_matrix
    )

    train_bilstm_model(bilstm_model, X_train_dl, y_train_dl, X_test_dl, y_test_dl, save_path="figures")
    evaluate_bilstm(
        model=bilstm_model,
        X_test=X_test_dl,
        y_test=y_test_dl,
        label_encoder=label_encoder,
        save_path="figures/bilstm_conf_matrix.png"
    )

    y_true_bilstm = np.argmax(y_test_dl, axis=1)
    y_pred_bilstm = np.argmax(bilstm_model.predict(X_test_dl), axis=1)
    label_names = label_encoder.classes_

    save_model_h5(bilstm_model, "models/bilstm_model.h5")
    joblib.dump(tokenizer, "models/tokenizer.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    # === Simpan ringkasan evaluasi ke file ===
    report_svm = classification_report(y_test_svm, svm_model.predict(X_test_svm))
    report_bilstm = classification_report(y_true_bilstm, y_pred_bilstm, target_names=label_names)

    with open("reports/performance_summary.txt", "w", encoding="utf-8") as f:
        f.write("=== Evaluasi SVM ===\n")
        f.write(report_svm + "\n\n")
        f.write("=== Evaluasi BiLSTM (FastText) ===\n")
        f.write(report_bilstm + "\n")

    print("\n✅ Pipeline selesai. Semua model, grafik dan laporan disimpan.")

if __name__ == "__main__":
    main()
