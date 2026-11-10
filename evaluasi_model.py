import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)

# ===================================
# KONFIGURASI (harus sama dengan train_model.py dan app.py)
# ===================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "hasil_evaluasi"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Pemakaian:
#   python evaluasi_model.py        -> evaluasi data test
#   python evaluasi_model.py val    -> evaluasi data val
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"

# ===================================
# MEMUAT MODEL DAN LABEL
# ===================================

print("Memuat model CNN BurnScan...")
model = tf.keras.models.load_model(MODEL_DIR / "burn_cnn.keras")

with open(MODEL_DIR / "labels.json", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

print("Kategori:", CLASS_NAMES)

# ===================================
# MEMBACA DATASET
# ===================================

print(f"Membaca dataset {SPLIT}...")
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / SPLIT,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,  # wajib False agar urutan label sejajar dengan prediksi
)


# Normalisasi sama persis dengan train_model.py: 0-255 -> 0-1
# (Rescaling ke -1..1 untuk MobileNetV2 sudah ada DI DALAM model)
def normalize_rgb(images, labels):
    return tf.cast(images, tf.float32) / 255.0, labels


dataset = dataset.map(normalize_rgb).prefetch(tf.data.AUTOTUNE)

# ===================================
# PREDIKSI
# ===================================

print(f"Melakukan prediksi data {SPLIT} BurnScan...")

y_true, y_pred = [], []
for images, labels in dataset:
    probs = model.predict(images, verbose=0)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(probs, axis=1).tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ===================================
# HASIL
# ===================================

akurasi = (y_true == y_pred).mean() * 100
qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    digits=2,
    zero_division=0,
)
matrix = confusion_matrix(
    y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
)

print("\n" + "=" * 50)
print(f"HASIL EVALUASI MODEL CNN BURNSCAN ({SPLIT.upper()})")
print("=" * 50)
print(f"Accuracy : {akurasi:.2f}%")
print(f"QWK      : {qwk:.3f}  (kesesuaian berurutan, 1 = sempurna)")
print("\nClassification Report:")
print(report)

print("Confusion matrix (baris = label asli, kolom = prediksi)")
print("           " + "  ".join(f"{n:>8s}" for n in CLASS_NAMES))
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:10s} " + "  ".join(f"{v:8d}" for v in matrix[i]))

print("\nAkurasi per kelas:")
for i, name in enumerate(CLASS_NAMES):
    total = matrix[i].sum()
    benar = matrix[i][i]
    persen = benar / total * 100 if total else 0
    print(f"  {name:8s}: {persen:5.1f}%  ({benar}/{total})")

# Seberapa sering salah ke tingkat tetangga (1 vs 2, 2 vs 3)
print("\nKesalahan terbesar per kelas:")
for i, name in enumerate(CLASS_NAMES):
    salah = matrix[i].copy()
    salah[i] = 0
    if salah.sum() > 0:
        j = int(np.argmax(salah))
        print(f"  {name} paling sering ditebak sebagai {CLASS_NAMES[j]} "
              f"({salah[j]} gambar)")

# ===================================
# MENYIMPAN HASIL
# ===================================

with open(OUTPUT_DIR / f"laporan_{SPLIT}.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {akurasi:.2f}%\n")
    f.write(f"QWK: {qwk:.3f}\n\n")
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(np.array2string(matrix))

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(matrix, cmap="Blues")
ax.set_xticks(range(len(CLASS_NAMES)))
ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Label asli")
ax.set_title(f"Confusion Matrix ({SPLIT})")
batas = matrix.max() / 2 if matrix.max() else 0
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        ax.text(
            j, i, matrix[i, j], ha="center", va="center",
            color="white" if matrix[i, j] > batas else "black",
        )
fig.colorbar(im)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f"confusion_matrix_{SPLIT}.png", dpi=150)
plt.close(fig)

print(f"\nHasil disimpan di folder {OUTPUT_DIR.name}.")
print("Evaluasi model BurnScan selesai.")