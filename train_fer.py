from ultralytics import YOLO
import os

def get_latest_model(runs_dir="runs/classify", base_model="yolov8n-cls.pt"):
    """
    runs/classify klasörüne bakar. En son oluşturulan train klasöründeki
    eğitilmiş modeli bulur. Eğer bulamazsa sıfır modele döner.
    """
    if not os.path.exists(runs_dir):
        return base_model
        
    # runs/classify içindeki train, train2 vb. klasörleri bul
    train_folders = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                     if d.startswith('train') and os.path.isdir(os.path.join(runs_dir, d))]
    
    if not train_folders:
        return base_model
        
    # Klasörleri oluşturulma (veya değiştirilme) tarihlerine göre en yeniden eskiye doğru sırala
    train_folders.sort(key=os.path.getctime, reverse=True)
    
    for folder in train_folders:
        # Önce last.pt'yi, yoksa best.pt'yi ara
        last_pt = os.path.join(folder, "weights", "last.pt")
        best_pt = os.path.join(folder, "weights", "best.pt")
        
        if os.path.exists(last_pt):
            return last_pt
        elif os.path.exists(best_pt):
            return best_pt
            
    return base_model

if __name__ == '__main__':
    # 1. Otomatik olarak en güncel modeli buluyoruz
    model_path = get_latest_model()
    
    if model_path == "yolov8n-cls.pt":
        print("Önceki bir eğitim bulunamadı. Sıfırdan başlanıyor (yolov8n-cls.pt) ...")
    else:
        print(f"Önceki eğitim bulundu! Kaldığı yerden üzerine ekleyerek eğitime devam edilecek: {model_path}")

    # Modeli yüklüyoruz
    model = YOLO(model_path)

    # İndirdiğimiz veri setinin absolute yolunu belirtiyoruz
    dataset_dir = os.path.abspath("fer2013_dataset")

    print(f"Model {dataset_dir} üzerinde eğitiliyor...")
    
    # Eğitim ayarlarını burada belirliyoruz. 
    if model_path == "yolov8n-cls.pt":
        # Sıfırdan eğitim başlatıyoruz
        model.train(
            data=dataset_dir, 
            epochs=50, 
            imgsz=48, 
            batch=64, 
            workers=0 
        )
    else:
        # Daha önceki bir eğitim bulunduysa, 'resume=True' diyerek 
        # öğrenme oranı (learning rate) vb. sıfırlanmadan TAM kaldığı yerden devam ediyoruz.
        # DİKKAT: resume=True olduğunda yeni bir 'train5' klasörü AÇMAZ. 
        # En son hangi klasörde (örn: train4) kaldıysa onun içine yazmaya devam eder!
        model.train(resume=True)
