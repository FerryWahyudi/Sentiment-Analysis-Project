import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.model_selection import StratifiedKFold

def train_svm(X_train, y_train, random_state=42, C=1.0):
    base_model = LinearSVC(random_state=random_state, C=C)
    model = CalibratedClassifierCV(estimator=base_model, cv=3)
    model.fit(X_train, y_train)
    return model

def evaluate_svm(model, X_test, y_test, save_dir="figures", show_plot=False):
    y_pred = model.predict(X_test)

    print("=== Classification Report (SVM) ===")
    report = classification_report(y_test, y_pred, output_dict=False)
    print(report)

    print("=== Confusion Matrix (SVM) ===")
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(list(set(y_test)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - SVM")

    os.makedirs(save_dir, exist_ok=True)
    cm_path = os.path.join(save_dir, "svm_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"[✔] Confusion matrix disimpan ke: {cm_path}")
    plt.close()

    # Optionally simpan classification report
    with open(os.path.join(save_dir, "svm_classification_report.txt"), "w") as f:
        f.write(report)

def cross_validate_svm(model, X, y, cv=5, save_dir="figures"):
    print("=== Cross Validation (SVM) ===")
    os.makedirs(save_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    reports = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = np.mean(y_pred == y_test)
        scores.append(acc)

        print(f"\n📊 Fold {i} Accuracy: {acc:.4f}")
        report = classification_report(y_test, y_pred, digits=4)
        print(report)
        reports.append(report)

        with open(os.path.join(save_dir, f"svm_fold{i}_report.txt"), "w") as f:
            f.write(report)

    # Rata-rata akurasi
    mean_score = np.mean(scores)
    print(f"\n✅ Mean Accuracy over {cv} folds: {mean_score:.4f}")

    # Simpan grafik
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, cv+1), scores, marker='o', label='Fold accuracy')
    plt.hlines(mean_score, 1, cv, colors='red', linestyles='dashed', label='Mean')
    plt.title(f"{cv}-Fold Cross Validation Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, cv+1))
    plt.legend()
    plt.grid(True)
    cv_plot_path = os.path.join(save_dir, "svm_cross_validation.png")
    plt.savefig(cv_plot_path, bbox_inches='tight')
    print(f"[✔] Grafik cross-validation disimpan ke: {cv_plot_path}")
    plt.close()

    # Simpan ringkasan skor ke file
    with open(os.path.join(save_dir, "svm_cross_val_summary.txt"), "w") as f:
        f.write("=== 5-Fold Cross Validation Scores ===\n")
        for i, score in enumerate(scores, 1):
            f.write(f"Fold {i}: {score:.4f}\n")
<<<<<<< HEAD
        f.write(f"\nMean Accuracy: {mean_score:.4f}\n")
=======
        f.write(f"\nMean Accuracy: {mean_score:.4f}\n")
>>>>>>> f886bbc (Add BiLSTM model using Git LFS)
