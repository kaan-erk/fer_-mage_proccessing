import os
from ultralytics import YOLO

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

def main():
    # Modelin kaydedildiği son klasörü otomatik buluyoruz.
    model_path = get_latest_model()
    
    if model_path == "yolov8n-cls.pt":
        print("Hata: Önceden eğitilmiş bir model bulunamadı!")
        return

    print(f"Eğitilmiş modelimiz ({model_path}) yükleniyor...")
    model = YOLO(model_path)

    # Kullanıcıdan test etmek istediği resmin yolunu iste
    print("\nÖrnek bir dosya yolu: fer2013_dataset/val/happy/PrivateTest_218533.jpg")
    print("fer2013_dataset/val/ klasörü içerisindeki diğer duygu klasörlerinden de resimler seçebilirsin.")
    test_image = input("Test etmek istediğiniz fotoğrafın dosya yolunu yapıştırın:\n> ").strip('"\'')
    
    if not os.path.exists(test_image):
        print(f"Hata: '{test_image}' yolunda bir resim bulunamadı.")
        return
    
    print(f"\n{test_image} resmi inceleniyor...")
    
    # Modelin resmi incelemesi ve tahminleri resim içine kaydederek predict klasörüne atması
    results = model.predict(source=test_image, save=True)
    
    # Classification model sonuç dökümünü almak:
    probs = results[0].probs
    best_class_index = probs.top1
    best_class_name = results[0].names[best_class_index]
    confidence = probs.top1conf.item()

    print("\n" + "="*50)
    print(f"TAHMİN EDİLEN DUYGU: {best_class_name.upper()} (Eminlik Oranı: %{confidence*100:.2f})")
    print("="*50)
    print("\nİşlem tamamlandı! Üzerine tahmin yazılmış resmi 'runs/classify/predict' (veya predict2, vb.) klasöründe bulabilirsin.")

if __name__ == "__main__":
    main()
