import joblib

def save_model_pickle(model, filename):
    joblib.dump(model, filename)
    print(f"[✔] SVM model saved to {filename}")

def save_model_h5(model, filename):
    model.save(filename)
    print(f"[✔] BiLSTM model saved to {filename}")
