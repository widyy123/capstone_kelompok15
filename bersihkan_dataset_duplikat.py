import hashlib
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "dataset"
BACKUP = BASE_DIR / "duplikat_dipindah"
EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Train diprioritaskan: gambar yang sama di val/test dipindahkan.
# Gambar tidak dihapus, hanya dipindah ke folder duplikat_dipindah/.
seen = {}
dipindah = 0

for split in ["train", "val", "test"]:
    for f in sorted((DATASET / split).rglob("*")):
        if not (f.is_file() and f.suffix.lower() in EXT):
            continue

        h = hashlib.md5(f.read_bytes()).hexdigest()

        if h in seen and seen[h] != split:
            tujuan = BACKUP / split / f.parent.name
            tujuan.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(tujuan / f.name))
            dipindah += 1
            print(f"Dipindah: {split}/{f.parent.name}/{f.name}")
        else:
            seen.setdefault(h, split)

print(f"\nTotal gambar duplikat dipindah: {dipindah}")
print("Jalankan ulang cek_dataset.py untuk memastikan sudah bersih.")