import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import joblib

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

def cross_validate_svm(model, X, y, cv=5):
    print("=== Cross Validation (SVM) ===")
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"Scores: {scores}")
    print(f"Mean Accuracy: {np.mean(scores):.4f}")

    # Plot CV score
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, cv+1), scores, marker='o', label='Fold accuracy')
    plt.hlines(np.mean(scores), 1, cv, colors='red', linestyles='dashed', label='Mean')
    plt.title(f"{cv}-Fold Cross Validation Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/svm_cross_validation.png", bbox_inches='tight')
    print("[✔] Grafik cross-validation disimpan ke: figures/svm_cross_validation.png")
    plt.close()
