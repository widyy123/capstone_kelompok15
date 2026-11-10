from pathlib import Path
import json
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ===================================
# KONFIGURASI
# ===================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"

IMAGE_SIZE = (224, 224)   # harus sama dengan IMAGE_SIZE di app.py
BATCH_SIZE = 32           # kalau out-of-memory, turunkan ke 16

EPOCHS_HEAD = 12
EPOCHS_FINE = 25
FINE_TUNE_LAYERS = 80     # sebelumnya 40; kalau hasil memburuk, kembalikan ke 40

# Pengali tambahan bobot Level_2 (kelas yang masih lemah)
LEVEL2_EXTRA_WEIGHT = 1.5

CLASS_NAMES = [
    "Level_1",
    "Level_2",
    "Level_3",
    "Normal"
]

tf.keras.utils.set_random_seed(42)

# Cadangkan model lama supaya tidak tertimpa
if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
    backup_dir = BASE_DIR / "model_lama"
    if not backup_dir.exists():
        shutil.copytree(MODEL_DIR, backup_dir)
        print(f"Model lama dicadangkan ke: {backup_dir}")

MODEL_DIR.mkdir(exist_ok=True)


# ===================================
# MEMBACA DATASET
# ===================================

def load_dataset(folder, shuffle=False):

    return tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42
    )


# Normalisasi RGB 0-255 -> 0-1 (sama dengan app.py)
def normalize_rgb(images, labels):

    return tf.cast(images, tf.float32) / 255.0, labels


def prepare_dataset(dataset):

    dataset = dataset.map(
        normalize_rgb,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


print("Membaca dataset training BurnScan...")
train_raw = load_dataset(DATASET_DIR / "train", shuffle=True)

print("Membaca dataset validasi BurnScan...")
val_raw = load_dataset(DATASET_DIR / "val")

print("Membaca dataset pengujian BurnScan...")
test_raw = load_dataset(DATASET_DIR / "test")

train_data = prepare_dataset(train_raw)
val_data = prepare_dataset(val_raw)
test_data = prepare_dataset(test_raw)


# ===================================
# BOBOT KELAS (MENGATASI DATA TIDAK SEIMBANG)
# ===================================

counts = []

for name in CLASS_NAMES:
    folder = DATASET_DIR / "train" / name
    counts.append(
        len([p for p in folder.iterdir() if p.is_file()])
    )

counts = np.array(counts, dtype=np.float32)
total = counts.sum()

class_weight = {
    i: float(total / (len(CLASS_NAMES) * counts[i]))
    for i in range(len(CLASS_NAMES))
}

# Tambahan bobot untuk Level_2
class_weight[CLASS_NAMES.index("Level_2")] *= LEVEL2_EXTRA_WEIGHT

print("\nJumlah gambar training per kelas:")
for name, count in zip(CLASS_NAMES, counts):
    print(f"  {name:8s}: {int(count)}")

print("Bobot kelas akhir:", {
    CLASS_NAMES[i]: round(w, 2) for i, w in class_weight.items()
})
print()


# ===================================
# ARSITEKTUR CNN (MobileNetV2)
# ===================================

print("Membangun arsitektur CNN BurnScan...")

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

# Augmentasi lebih kuat (hanya aktif saat training)
augment = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.25, value_range=(0.0, 1.0)),
    layers.RandomContrast(0.25),
], name="augmentasi")

inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = augment(inputs)

# Input aplikasi 0-1, MobileNetV2 butuh -1..1.
x = layers.Rescaling(scale=2.0, offset=-1.0)(x)

x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation="softmax")(x)

model = models.Model(inputs, outputs)


def compile_model(learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


def make_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]


# ===================================
# TAHAP 1: LATIH LAPISAN KLASIFIKASI
# ===================================

compile_model(1e-3)
model.summary()

print("\nTahap 1: melatih lapisan klasifikasi BurnScan...")

history_head = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_HEAD,
    class_weight=class_weight,
    callbacks=make_callbacks()
)


# ===================================
# TAHAP 2: FINE-TUNING
# ===================================

print("\nTahap 2: fine-tuning backbone BurnScan...")

base.trainable = True

for layer in base.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False

compile_model(3e-5)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE,
    class_weight=class_weight,
    callbacks=make_callbacks()
)


# ===================================
# EVALUASI MODEL
# ===================================

print("\nMelakukan evaluasi data test BurnScan...")

test_loss, test_accuracy = model.evaluate(test_data)

print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

y_true, y_pred = [], []

for images, labels in test_data:
    probs = model.predict(images, verbose=0)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(probs, axis=1).tolist())

matrix = np.zeros(
    (len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int
)

for t, p in zip(y_true, y_pred):
    matrix[t][p] += 1

print("\nConfusion matrix (baris = label asli, kolom = prediksi)")
print("           " + "  ".join(f"{n:>8s}" for n in CLASS_NAMES))

for i, name in enumerate(CLASS_NAMES):
    print(
        f"{name:10s} "
        + "  ".join(f"{v:8d}" for v in matrix[i])
    )

print("\nAkurasi per kelas:")
for i, name in enumerate(CLASS_NAMES):
    total_kelas = matrix[i].sum()
    benar = matrix[i][i]
    persen = (benar / total_kelas * 100) if total_kelas else 0
    print(f"  {name:8s}: {persen:5.1f}%  ({benar}/{total_kelas})")

# Daftar gambar Level_2 yang salah prediksi (untuk dicek manual labelnya)
# test_raw tidak di-shuffle, jadi urutan file_paths sama dengan y_true/y_pred
level2_index = CLASS_NAMES.index("Level_2")

print("\nGambar Level_2 yang salah prediksi:")
ada_salah = False
for path, t, p in zip(test_raw.file_paths, y_true, y_pred):
    if t == level2_index and p != level2_index:
        ada_salah = True
        print(f"  {Path(path).name} -> diprediksi {CLASS_NAMES[p]}")
if not ada_salah:
    print("  (tidak ada)")


# ===================================
# MENYIMPAN MODEL, LABEL, DAN RIWAYAT
# ===================================

model.save(MODEL_DIR / "burn_cnn.keras")

with open(
    MODEL_DIR / "labels.json", "w", encoding="utf-8"
) as file:
    json.dump(
        CLASS_NAMES, file,
        ensure_ascii=False, indent=4
    )

history = {}

for key in history_head.history:
    history[key] = (
        history_head.history[key]
        + history_fine.history.get(key, [])
    )

with open(
    MODEL_DIR / "training_history.json", "w", encoding="utf-8"
) as file:
    json.dump(
        {k: [float(v) for v in vals] for k, vals in history.items()},
        file, indent=4
    )

print("\nPelatihan CNN BurnScan selesai.")
print("Model berhasil disimpan di folder model.")