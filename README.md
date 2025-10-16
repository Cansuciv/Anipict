# 🐐 Keçi Türleri & Böcek/Sürüngen Tespiti – YOLOv8 / YOLOv11 Projesi

Bu proje, görüntüler üzerinde hem image classification (resim sınıflandırma) hem de object detection (nesne tespiti) tekniklerini kullanarak iki temel alanda çalışır:

Keçi Türleri Tespiti (YOLOv8l-cls, YOLOv11l, YOLOv11n)

Zararlı Böcekler, Sürüngenler, Akrepler, Yengeçler gibi canlıların tespiti (YOLOv11n)

## 📁 Proje İçeriği

### 1. Keçi Türleri – Image Classification (YOLOv8l-cls)

- **Model**: YOLOv8l-cls
- **Veri Seti**: 2.480 etiketli görüntü
- **Etiketleme**: `makesense.ai`
- **Veri Artırma**: Roboflow ile yapıldı
- **Eğitim Parametreleri**:
  - Epoch: 100
  - Görüntü boyutu: 640x640
  - Batch size: 8
  - Workers: 8
- **Donanım**: Google Colab (GPU)

#### Sınıflandırılan Keçi Türleri (Toplam 9 Tür)
- Ankara Keçisi
- Halep Keçisi
- Honamlı Keçisi
- Kilis Keçisi
- Kıl Keçisi
- Malta Keçisi
- Norduz Keçisi
- Saanen Keçisi
- Yaban Keçisi
- Ayrıca, **Halep** ve **Yaban keçisi** birlikte bulunan 17 görsel mevcut.


<!-- Görsel örneği --> 
![Keçi Türleri Image Classification](./ReadmeResim/kecilerImageClassification.png)

---

### 2. Keçi Türleri – Object Detection

#### A) YOLOv11l
- Daha doğru ama daha yavaş
- Kullanılan model: `YOLOv11l`
- Epoch: 100, Görüntü boyutu: 640x640
- Batch size: 8, Workers: 8
- Donanım: Google Colab


<!-- Detected output örneği -->
![Keçi Türleri YOLOv11l](./ReadmeResim/KeciYolov11l.png)



#### B) YOLOv11n
- Daha hafif ve hızlı model
- Kullanılan model: `YOLOv11n`
- Epoch: 100, Görüntü boyutu: 640x640
- Batch size: 8, Workers: 8
- Donanım: Google Colab
- Video üzerinden tahminlerde `ONNX` dönüştürmesi yapıldı

#### Veri Seti
Toplam Görüntü: 10.517 (veri artırımı sonrası)
Veri artırımı sayesinde tüm keçi türleri için dengeli örnek sayısı sağlanmıştır:
- Ankara Keçisi: 1023
- Halep Keçisi: 765
- Honamlı Keçisi: 814
- Kilis Keçisi: 940
- Kıl Keçisi: 1275
- Malta Keçisi: 810
- Norduz Keçisi: 702
- Saanen Keçisi: 887
- Yaban Keçisi: 3301


#### Performans (YOLOv11n)
###### 1️⃣ Train Loss
- **train/box_loss** = 1.05399 → Modelin kutu (bounding box) tahmin hatası  
- **train/cls_loss** = 1.02224 → Modelin sınıf tahmin hatası  
- **train/dfl_loss** = 1.50112 → Distribution Focal Loss, kutu konumlandırma doğruluğunu ölçer  

**Toplam train loss** ≈ 3.57735 → Eğitimin toplam kaybı

###### 2️⃣ Validation Loss
- **val/box_loss** = 1.26823 → Doğrulama setinde kutu tahmin hatası  
- **val/cls_loss** = 0.84737 → Doğrulama setinde sınıf tahmin hatası  
- **val/dfl_loss** = 1.68639 → Doğrulama setinde kutu konumlandırma hatası  

**Toplam val loss** ≈ 3.80199 → Modelin doğrulama performansı toplam kaybı

###### 3️⃣ Precision / Recall / F1 (Validation)
- **Precision** = 0.79162 (79.2%) → Modelin tespit ettiği kutuların %79.2’si doğru  
- **Recall** = 0.74188 (74.2%) → Gerçek nesnelerin %74.2’si model tarafından tespit edildi  

**F1-score** ≈ 0.7666 (76.7%) → Precision ve Recall dengesi

###### 4️⃣ mAP değerleri
- **mAP@0.5** = 0.80067 (80.1%) → IoU≥0.5 olduğunda ortalama başarı  
- **mAP@0.5:0.95** = 0.53310 (53.3%) → Daha sıkı IoU eşiklerinde ortalama başarı (konumlandırma doğruluğu)

###### 5️⃣ Learning Rate (epoch 55)
- **lr/pg0** = 4.10001e-05  
- **lr/pg1** = 4.10001e-05  
- **lr/pg2** = 4.10001e-05 → Modelin ağırlıklarını güncelleme hızı

<!-- Detected output örneği -->
![Keçi Türleri YOLOv11l](./ReadmeResim/KeciYolov11n.png)

---

### 3. Böcek, Sürüngen ve Akrep Tespiti (YOLOv11n)

- 22 farklı tür üzerinde nesne tespiti yapılmıştır
- Toplam görüntü sayısı: 31.195  
- Veri setleri:  Roboflow’dan 7 farklı kaynaktan veri seti bulunmuştur. Ek olarak “Akdeniz Münzevi Örümceği, Anadolu Sarı Akrebi, Kara Akrep”türlerini içeren bir veri seti hazırlandı. Sonra toplam bu 8 veri seti birleştirildi. Hazır alınan veri setleri:
  - https://universe.roboflow.com/project-lnrc3/zararli-bocekler 
  - https://universe.roboflow.com/4702/dangerous-farm-insects-dataset 
  - https://universe.roboflow.com/insects-tibsl/insects-vtdmw 
  - https://universe.roboflow.com/nirmani/yolo-custome 
  - https://universe.roboflow.com/dave-ellomar-jamilla-qzlwc/crab-classification-4 
  - https://universe.roboflow.com/rolex-tgyvd/scorpion-detection1 
  - https://universe.roboflow.com/crab-x2izg/my-first-project-1ev5n 
- Eğitilen türlerden bazıları:
    - Akdeniz Munzevi Orumcegi - 3439
    - Anadolu Sari Akrebi - 4610
    - Kara Akrep - 2893
    - Katil Ari - 1076
    - Yaprak Biti - 1376
    - Kahverengi Kokarca Bocegi - 1350
    - Lahana Tittillari - 900
    - Patates Bocegi - 1023
    - Misir Kurdu - 1044
    - Misir Yuvarlak Kurdu - 1035
    - Sonbahar Ordu Kurdu - 1023
    - Sirke Sinegi - 1129
    - Kum Yengeci - 728
    - uc benekli yuzuen yengec - 688
    - Kirmizi Orumcek - 1717
    - Trips - 1092
    - Mavi Yengec - 742
    - Kemanci Yengec - 1045
    - baklagil kabarcik bocegi - 1035
    - Camur Yengeci - 1020
    - Pirinc Gal Sinegi - 1000
    - Beyaz Sirtli Bitki Zararlisi - 1230


#### Performans
###### 1️⃣ Train Loss
- **train/box_loss = 0.93814** → Modelin kutu (bounding box) konumlandırma hatası
- **train/cls_loss = 0.97813** → Modelin sınıf tahmin hatası
- **train/dfl_loss = 1.40131** → Distribution Focal Loss; kutu sınırlarının doğruluk hassasiyetini ölçer
- **Toplam train loss ≈ 3.31758** → Modelin eğitim sürecindeki genel hata miktarı

###### 2️⃣ Validation Loss
- **val/box_loss = 0.91852** → Doğrulama setinde kutu tahmin hatası
- **val/cls_loss = 0.46450** → Doğrulama setinde sınıf tahmin hatası
- **val/dfl_loss = 1.41069** → Kutu konumlandırma doğruluğu hatası
- **Toplam val loss ≈ 2.79371** → Modelin doğrulama performansında toplam hata

###### 3️⃣ Precision / Recall / F1 (Doğrulama Başarımı)
- **Precision = 0.93523 (93.5%)** → Tespit edilen nesnelerin %93.5’i doğru
- **Recall = 0.97060 (97.1%)** → Gerçek nesnelerin %97.1’i model tarafından yakalanıyor
- **F1-score ≈ 0.9526 (95.3%)** → Precision ve Recall arasındaki dengeyi gösterir

###### 4️⃣ mAP Değerleri (Ortalama Doğruluk)
- **mAP@0.5 = 0.97305 (97.3%)** → IoU≥0.5 eşik değerinde genel tespit doğruluğu
- **mAP@0.5:0.95 = 0.73493 (73.5%)** → Daha sıkı IoU aralıklarında ortalama başarı

###### 5️⃣ Learning Rate (Öğrenme Oranı)
- **lr/pg0 = 0.0069031**
- **lr/pg1 = 0.0069031**
- **lr/pg2 = 0.0069031** → Modelin son epoch’larda ince ayar yaptığı öğrenme oranı

<!-- Detected akrep örneği -->
![Akrep Tespiti YOLOv11l](./ReadmeResim/SurungenBocek.png)

---


## 🛠️ Kullanılan Araçlar ve Teknolojiler

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [YOLOv11](https://github.com/WongKinYiu/yolov11)
- Google Colab & Kaggle (GPU desteği)
- Roboflow (veri artırma ve dönüştürme)
- Makesense.ai (etiketleme)
- OpenCV & ONNX Runtime (video üzerinden tespit)



