from pathlib import Path
from io import BytesIO
import base64
import json
import os

import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps, UnidentifiedImageError

from flask import (
    Flask,
    render_template,
    request
)

# ==========================================
# KONFIGURASI APLIKASI
# ==========================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "burn_cnn.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

IMAGE_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "webp"
}

EXPECTED_CLASSES = [
    "Level_1",
    "Level_2",
    "Level_3",
    "Normal"
]

DISPLAY_NAMES = {
    "Level_1": "Level 1",
    "Level_2": "Level 2",
    "Level_3": "Level 3",
    "Normal": "Kulit Normal"
}

# ------------------------------------------
# Ambang keputusan (boleh disesuaikan setelah diuji)
# ------------------------------------------
MIN_CONFIDENCE = 0.60      # probabilitas tertinggi minimal 60%
MIN_MARGIN = 0.15          # selisih kelas 1 dan 2 minimal 15%
LEVEL3_ALERT = 0.25        # peringatan jika Level_3 >= 25% walau bukan teratas

# Ambang kualitas foto
MIN_SIDE = 150             # sisi terpendek foto (piksel)
MIN_BRIGHTNESS = 40        # rata-rata kecerahan (0-255)
MAX_BRIGHTNESS = 235
MIN_SHARPNESS = 60.0       # variansi Laplacian; makin kecil makin buram


# ==========================================
# MEMUAT MODEL CNN
# ==========================================

model = None
class_names = []

def load_cnn_model():
    global model, class_names, IMAGE_SIZE

    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print("Model CNN atau labels.json belum ditemukan.")
        return

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as file:
            labels = json.load(file)

        if labels != EXPECTED_CLASSES:
            raise ValueError(f"Urutan label tidak sesuai: {labels}")

        loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        if loaded_model.output_shape[-1] != len(labels):
            raise ValueError("Jumlah output model tidak cocok.")

        height, width = loaded_model.input_shape[1:3]
        IMAGE_SIZE = (int(height), int(width))

        model = loaded_model
        class_names = labels

        jenis = (
            "MobileNetV2"
            if any("mobilenet" in layer.name.lower() for layer in model.layers)
            else "CNN buatan sendiri"
        )

        print("=" * 45)
        print("MODEL CNN BERHASIL DIMUAT (BurnScan)")
        print("Jenis model :", jenis)
        print("Ukuran input:", IMAGE_SIZE)
        print("Kategori    :", class_names)
        print("=" * 45)

    except Exception as error:
        print("Gagal memuat CNN:", error)
        model = None
        class_names = []


# ==========================================
# VALIDASI & PREVIEW FOTO
# ==========================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def buat_preview(image):
    preview = image.copy()
    preview.thumbnail((600, 600))

    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=85)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def cek_kualitas_foto(image):
    """Mengembalikan daftar peringatan tentang kualitas foto."""
    peringatan = []

    if min(image.size) < MIN_SIDE:
        peringatan.append(
            "Resolusi foto terlalu kecil. Gunakan foto yang lebih jelas."
        )

    gray = np.asarray(image.convert("L"), dtype=np.float32)

    kecerahan = float(gray.mean())
    if kecerahan < MIN_BRIGHTNESS:
        peringatan.append(
            "Foto terlalu gelap. Ulangi dengan pencahayaan yang lebih terang."
        )
    elif kecerahan > MAX_BRIGHTNESS:
        peringatan.append(
            "Foto terlalu terang atau silau. Hindari cahaya langsung."
        )

    # Ketajaman: variansi Laplacian (tanpa OpenCV)
    if gray.shape[0] > 2 and gray.shape[1] > 2:
        lap = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1] + gray[2:, 1:-1]
            + gray[1:-1, :-2] + gray[1:-1, 2:]
        )
        if float(lap.var()) < MIN_SHARPNESS:
            peringatan.append(
                "Foto tampak buram. Dekatkan kamera dan pastikan fokus."
            )

    return peringatan


# ==========================================
# PREPROCESSING & PREDIKSI CNN
# ==========================================

def preprocess_for_cnn(image):
    # Sama dengan train_model.py: RGB 0-255 -> 0-1.
    # Rescaling ke -1..1 untuk MobileNetV2 sudah ada di dalam model.
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    rgb_tensor = tf.convert_to_tensor(rgb)

    resized = tf.image.resize(rgb_tensor, IMAGE_SIZE, method="bilinear")

    return tf.expand_dims(resized, axis=0)

def prediksi_cnn(image):
    if model is None:
        return None

    input_tensor = preprocess_for_cnn(image)
    probabilities = model.predict(input_tensor, verbose=0)[0]

    # Log ke terminal untuk diagnosis
    print("-" * 40)
    for nama, p in zip(class_names, probabilities):
        print(f"{nama:8s}: {float(p) * 100:6.2f}%")
    print("-" * 40)

    urut = np.argsort(probabilities)[::-1]
    top1, top2 = int(urut[0]), int(urut[1])
    p1, p2 = float(probabilities[top1]), float(probabilities[top2])

    label = class_names[top1]
    confidence = p1 * 100

    # Status keputusan
    yakin = (p1 >= MIN_CONFIDENCE) and ((p1 - p2) >= MIN_MARGIN)

    peringatan = []

    idx_l3 = class_names.index("Level_3")
    if label != "Level_3" and float(probabilities[idx_l3]) >= LEVEL3_ALERT:
        peringatan.append(
            "Ada kemungkinan luka berat. Jika Anda ragu atau ada gejala "
            "serius, segera periksakan ke fasilitas kesehatan."
        )

    # Daftar probabilitas semua kelas untuk ditampilkan
    semua = [
        {
            "label": nama,
            "display": DISPLAY_NAMES.get(nama, nama),
            "persen": round(float(p) * 100, 2),
        }
        for nama, p in zip(class_names, probabilities)
    ]

    return {
        "label": label,
        "display": DISPLAY_NAMES.get(label, label),
        "confidence": round(confidence, 2),
        "yakin": yakin,
        "semua": semua,
        "peringatan": peringatan,
    }


# ==========================================
# ROUTING HALAMAN
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tentang")
def tentang():
    return render_template("tentang.html")

@app.route("/edukasi")
def edukasi():
    return render_template("edukasi.html")

@app.route("/deteksi", methods=["GET", "POST"])
def deteksi():
    preview = None
    error = None
    result = None
    kualitas = []

    if request.method == "POST":
        foto = request.files.get("foto")

        if foto is None or not foto.filename:
            error = "Silakan pilih foto terlebih dahulu."
        elif not allowed_file(foto.filename):
            error = "Format harus JPG, JPEG, PNG, atau WEBP."
        else:
            try:
                image = Image.open(foto.stream)
                image.verify()

                foto.stream.seek(0)
                image = Image.open(foto.stream)

                if image.width * image.height > 20_000_000:
                    raise ValueError("Resolusi gambar terlalu besar.")

                image.load()
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")

                preview = buat_preview(image)
                kualitas = cek_kualitas_foto(image)

                if model is not None:
                    result = prediksi_cnn(image)
                else:
                    error = "Model CNN belum tersedia. Silakan hubungi admin."

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                tf.errors.OpError
            ) as exc:
                print("Kesalahan:", exc)
                error = "Gambar tidak valid atau gagal diproses."

    return render_template(
        "deteksi.html",
        preview=preview,
        error=error,
        result=result,
        kualitas=kualitas,
        model_ready=model is not None
    )


# ==========================================
# ERROR HANDLER & SERVER
# ==========================================

@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "deteksi.html",
        preview=None,
        error="Ukuran foto maksimal 5 MB.",
        result=None,
        kualitas=[],
        model_ready=model is not None
    ), 413

load_cnn_model()

if __name__ == "__main__":
    # Debug hanya aktif jika diminta: set FLASK_DEBUG=1
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")