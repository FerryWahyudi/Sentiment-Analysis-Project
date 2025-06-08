import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def get_tfidf_features(texts, max_features=5000, return_vectorizer=False):
    tfidf = TfidfVectorizer(max_features=max_features)
    X_tfidf = tfidf.fit_transform(texts)

    if return_vectorizer:
        return X_tfidf, tfidf
    return X_tfidf


def get_bilstm_features(texts, labels, vocab_size=10000, maxlen=100):
    # Tokenisasi
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

    # Encode label
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded)

    return padded, labels_onehot, tokenizer, le

def load_fasttext_embeddings(filepath, tokenizer, embedding_dim=300):
    print(f"📥 Memuat FastText dari {filepath}...")
    embeddings_index = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split()
            word = values[0]
            try:
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = vector
            except:
                continue  # abaikan baris rusak

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if i >= vocab_size:
            continue
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector

    return embedding_matrix
