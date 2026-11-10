import json
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
LABELS_PATH = BASE_DIR / "model" / "labels.json"

FOLDERS = ["train", "val", "test"]
CLASSES = sorted(p.name for p in (DATASET_DIR / "train").iterdir() if p.is_dir())
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

jumlah = defaultdict(dict)      # jumlah[folder][kelas] = gambar valid
rusak_list = []                 # daftar gambar rusak
hash_map = defaultdict(list)    # hash -> [(folder, kelas, nama file)]
ukuran = set()                  # ukuran gambar yang ditemukan

print("=" * 60)
print("PEMERIKSAAN DATASET LUKA BAKAR")
print("=" * 60)

# ---------------------------------------------------------------
# 1. Scan semua gambar
# ---------------------------------------------------------------
for folder in FOLDERS:
    for kelas in CLASSES:
        path = DATASET_DIR / folder / kelas

        if not path.exists():
            print(f"[!] {folder}/{kelas}: FOLDER TIDAK DITEMUKAN")
            jumlah[folder][kelas] = 0
            continue

        valid = 0
        for file in path.iterdir():
            if not (file.is_file() and file.suffix.lower() in EXTENSIONS):
                continue
            try:
                with Image.open(file) as img:
                    ukuran.add(img.size)
                    img.verify()
                valid += 1
                h = hashlib.md5(file.read_bytes()).hexdigest()
                hash_map[h].append((folder, kelas, file.name))
            except Exception:
                rusak_list.append(f"{folder}/{kelas}/{file.name}")

        jumlah[folder][kelas] = valid

# ---------------------------------------------------------------
# 2. Ringkasan per split dan per kelas
# ---------------------------------------------------------------
for folder in FOLDERS:
    total = sum(jumlah[folder].values())
    print(f"\nDATASET: {folder.upper()} (total {total})")
    print("-" * 60)
    for kelas in CLASSES:
        n = jumlah[folder].get(kelas, 0)
        persen = n / total * 100 if total else 0
        print(f"  {kelas:10s}: {n:5d} gambar ({persen:5.1f}%)")

# Total per kelas gabungan semua split
print("\nTOTAL PER KELAS (train + val + test)")
print("-" * 60)
total_semua = sum(sum(jumlah[f].values()) for f in FOLDERS)
for kelas in CLASSES:
    n = sum(jumlah[f].get(kelas, 0) for f in FOLDERS)
    persen = n / total_semua * 100 if total_semua else 0
    print(f"  {kelas:10s}: {n:5d} gambar ({persen:5.1f}%)")

# ---------------------------------------------------------------
# 3. Peringatan ketidakseimbangan kelas dan data val/test terlalu sedikit
# ---------------------------------------------------------------
print("\nANALISIS")
print("-" * 60)
train_counts = [jumlah["train"].get(k, 0) for k in CLASSES]
if train_counts and max(train_counts) > 0:
    min_n, max_n = min(train_counts), max(train_counts)
    rasio = max_n / max(min_n, 1)
    if rasio >= 2:
        print(f"[!] Train tidak seimbang: kelas terbesar {rasio:.1f}x kelas terkecil.")
        print("    Gunakan class_weight dan augmentasi saat training.")
    else:
        print("[OK] Distribusi kelas di train cukup seimbang.")

for folder in ["val", "test"]:
    for kelas in CLASSES:
        n = jumlah[folder].get(kelas, 0)
        if n < 30:
            print(f"[!] {folder}/{kelas} hanya {n} gambar: hasil evaluasi bisa tidak stabil.")

# ---------------------------------------------------------------
# 4. Duplikat antar split (data leakage)
# ---------------------------------------------------------------
print("\nDUPLIKAT ANTAR SPLIT")
print("-" * 60)
ada_duplikat = False
for h, lst in hash_map.items():
    if len(lst) > 1 and len({x[0] for x in lst}) > 1:
        ada_duplikat = True
        print("  [!] Duplikat:")
        for folder, kelas, nama in lst:
            print(f"      {folder}/{kelas}/{nama}")
if not ada_duplikat:
    print("  [OK] Tidak ada duplikat antar split.")

# Duplikat yang sama tapi beda kelas (label konflik)
print("\nLABEL KONFLIK (gambar sama, kelas berbeda)")
print("-" * 60)
ada_konflik = False
for h, lst in hash_map.items():
    if len({x[1] for x in lst}) > 1:
        ada_konflik = True
        print("  [!] Konflik:")
        for folder, kelas, nama in lst:
            print(f"      {folder}/{kelas}/{nama}")
if not ada_konflik:
    print("  [OK] Tidak ada label konflik.")

# ---------------------------------------------------------------
# 5. Gambar rusak
# ---------------------------------------------------------------
print("\nGAMBAR RUSAK")
print("-" * 60)
if rusak_list:
    for r in rusak_list:
        print(f"  [!] {r}")
else:
    print("  [OK] Tidak ada gambar rusak.")

# ---------------------------------------------------------------
# 6. Ukuran gambar
# ---------------------------------------------------------------
print(f"\nUkuran gambar berbeda: {len(ukuran)}")
if len(ukuran) > 1:
    print("  Pastikan semua di-resize ke ukuran yang sama saat training "
          "dan saat prediksi di app.py.")

# ---------------------------------------------------------------
# 7. Urutan label (Keras mengurutkan folder secara alfabet)
# ---------------------------------------------------------------
print("\nURUTAN LABEL")
print("-" * 60)
urutan_keras = CLASSES
print("  Urutan indeks Keras (alfabet):")
for i, k in enumerate(urutan_keras):
    print(f"    {i} = {k}")

if LABELS_PATH.exists():
    try:
        with open(LABELS_PATH, encoding="utf-8") as f:
            labels = json.load(f)
        print(f"  Isi labels.json: {labels}")
        print("  Pastikan urutannya sama dengan urutan indeks Keras di atas.")
    except Exception as e:
        print(f"  [!] Gagal membaca labels.json: {e}")
else:
    print("  labels.json tidak ditemukan di folder model/.")

# ---------------------------------------------------------------
# Ringkasan akhir
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print(f"TOTAL GAMBAR VALID : {total_semua}")
print(f"TOTAL GAMBAR RUSAK : {len(rusak_list)}")
print("=" * 60)
print("\nPemeriksaan dataset selesai.")