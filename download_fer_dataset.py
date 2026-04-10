import os
import kaggle
import shutil

print("Kaggle ile bağlantı kuruluyor...")
kaggle.api.authenticate()

DATASET_NAME = "msambare/fer2013"
DOWNLOAD_PATH = "fer2013_dataset"

print(f"'{DATASET_NAME}' veri seti '{DOWNLOAD_PATH}' klasörüne indiriliyor...")
kaggle.api.dataset_download_files(DATASET_NAME, path=DOWNLOAD_PATH, unzip=True)
print("İndirme ve çıkartma işlemi tamamlandı!")

# Klasörleri YOLOv8 classification'a uyarlıyoruz
test_path = os.path.join(DOWNLOAD_PATH, "test")
val_path = os.path.join(DOWNLOAD_PATH, "val")

if os.path.exists(test_path) and not os.path.exists(val_path):
    os.rename(test_path, val_path)
    print("YOLOv8 standartları için 'test' klasörü 'val' olarak adlandırıldı.")
