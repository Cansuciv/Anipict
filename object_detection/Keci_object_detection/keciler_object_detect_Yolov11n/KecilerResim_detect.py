from ultralytics import YOLO
import cv2

# Modeli yükle
model = YOLO("keciler_best.pt")

# Görseli oku (yolunu buraya gir)
# data\\keciler_dataset\\train\\YabanKecisi\\YabanKecisi1_jpg.rf.231bbb22c9cf118ab5ea1079cc1d3089.jpg
# data\\keciler_dataset\\train\\SaanenKecisi\\SaanenKecisi13_jpg.rf.9e0e4d2ce17f59ca962edc2fc5103808.jpg
# data\\keciler_dataset\\train\\YabanKecisi+HalepKecisi\\YabanKecisi96_jpg.rf.fc0a157e765685309c8e22d9bc3b8bd1.jpg
# data\\keciler_dataset\\train\\NorduzKecisi\\NorduzKecisi2_jpg.rf.5b7847e3498fd3ab3786f5a19374580a.jpg
# data\\keciler_dataset\\train\\MaltaKecisi\\MaltaKecisi4_jpg.rf.7c43e64fb99f55733e0acbd7512d0348.jpg
# data\\keciler_dataset\\train\\Kilkecisi\\KilKecisi34_jpg.rf.d64c11843a6996528e93dabbcc1ff9d9.jpg
# data\\keciler_dataset\\train\\KilisKecisi\\KilisKecisi13_jpg.rf.d70e92660cae05809d55ad9abc55aa24.jpg
# data\\keciler_dataset\\train\\HonamliKecisi\\HonamliKecisi29_jpg.rf.9d388b02f617d0825d9cc06d7d9392a9.jpg
# data\\keciler_dataset\\train\\HalepKecisi\\HalepKecisi39_jpg.rf.ae0979510a07e41df899a5578c669e56.jpg
# data\\keciler_dataset\\train\\AnkaraKecisi\\AnkaraKecisi43_jpg.rf.9bf1b1f803506e5f965cf940b336e167.jpg

image_path = "data\\keciler_dataset\\train\\YabanKecisi\\YabanKecisi1_jpg.rf.231bbb22c9cf118ab5ea1079cc1d3089.jpg"
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
