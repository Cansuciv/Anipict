from ultralytics import YOLO
import cv2

# Modeli yükle
model = YOLO("SurungenBocek_best.pt")

# Görseli oku (yolunu buraya gir)
# AkdenizMunzeviOrumcegi65_jpg.rf.8f341739544a7bb9e2af8c9a17f99b2e.jpg
# AnadoluSariAkrebi106_jpg.rf.3e093255772d491578d97cd0ba0dded8.jpg
# ZehirliKaraAkrep255_jpg.rf.4f102032aeac29bf4d877b7e065679a3.jpg

image_path = "data\\SurungenBocekDataset\\test\\images\\ZehirliKaraAkrep255_jpg.rf.4f102032aeac29bf4d877b7e065679a3.jpg"

image = cv2.imread(image_path)

# Tahmin yap
results = model.predict(image, imgsz=640, conf=0.3)

# Algılanan sınıfları yazdır (isteğe bağlı)
if len(results[0]) > 0:
    print("Algılanan sınıflar:", results[0].boxes.cls)

# Tahmin sonuçlarını görsele uygula ve göster
annotated_image = results[0].plot()
cv2.imshow("YOLO Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
