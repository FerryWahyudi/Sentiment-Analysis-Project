from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import os

def build_bilstm_model(vocab_size, embedding_dim, input_length, embedding_matrix=None):
    model = Sequential()
    if embedding_matrix is not None:
        model.add(Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            input_length=input_length,
                            trainable=False))
    else:
        model.add(Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            input_length=input_length))

    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))  # 3 kelas: positif, netral, negatif
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_bilstm_model(model, X_train, y_train, X_val, y_val, save_path=None):
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Plot grafik loss dan akurasi
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, "BiLSTM_loss_accuracy.png")
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"[✔] Grafik loss/accuracy disimpan ke {plot_path}")
        plt.close()
    else:
        plt.show()

def evaluate_bilstm(model, X_test, y_test, label_encoder=None, save_path=None):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"=== Evaluation BiLSTM ===\nLoss: {loss:.4f} - Accuracy: {acc:.4f}")

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    if label_encoder:
        target_names = label_encoder.classes_
    else:
        target_names = [str(i) for i in range(y_test.shape[1])]

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - BiLSTM")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[✔] Confusion matrix disimpan ke {save_path}")
        plt.close()
    else:
        plt.show()
