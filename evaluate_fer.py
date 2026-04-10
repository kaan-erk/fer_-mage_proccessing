from ultralytics import YOLO
import os

def get_latest_model(runs_dir="runs/classify", base_model="yolov8n-cls.pt"):
    if not os.path.exists(runs_dir):
        return base_model
        
    train_folders = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                     if d.startswith('train') and os.path.isdir(os.path.join(runs_dir, d))]
    
    if not train_folders:
        return base_model
        
    train_folders.sort(key=os.path.getctime, reverse=True)
    
    for folder in train_folders:
        best_pt = os.path.join(folder, "weights", "best.pt")
        last_pt = os.path.join(folder, "weights", "last.pt")
        
        if os.path.exists(best_pt):
            return best_pt
        elif os.path.exists(last_pt):
            return last_pt
            
    return base_model

if __name__ == '__main__':
    # Otomatik son modeli bul
    model_path = get_latest_model()
    print(f"Değerlendirilecek model: {model_path}")
    
    model = YOLO(model_path)

    dataset_dir = os.path.abspath("fer2013_dataset")
    
    print("Model test verisi (val) üzerinde değerlendiriliyor (validation)...")
    
    # Burada model, doğrulama (val/test) seti üzerinde performansını ölçecektir.
    metrics = model.val(data=dataset_dir)
    
    # Metriklerden accuracy (doğruluk) hesaplarını çıkartalım:
    top1 = metrics.top1
    top5 = metrics.top5
    
    print("\n" + "="*50)
    print("DOĞRULUK (ACCURACY) METRİKLERİ:")
    print(f"  Top-1 Doğruluk (Doğru Sınıfı 1. Sırada Bulma): {top1 * 100:.2f}%")
    print(f"  Top-5 Doğruluk (Doğru Sınıfı İlk 5'te Bulma): {top5 * 100:.2f}%")
    print("="*50)
    print("\nDaha fazla çıktı ve görselleri 'runs/classify/val' konumunda bulabilirsiniz.")
