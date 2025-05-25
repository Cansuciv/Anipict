from ultralytics import YOLO
import cv2

# Modeli yükle (kendi eğittiğiniz model veya resmi YOLO modeli)
model = YOLO("best.pt")  # "yolov8n.pt" veya "yolov9c.pt" de kullanabilirsiniz

# Webcam'i aç
cap = cv2.VideoCapture(0)  # 0 = varsayılan kamera

while True:
    # Kameradan görüntü al
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO ile tespit yap
    results = model.predict(frame, imgsz=640, conf=0.5)

    # Sonuçları görüntüye işle
    annotated_frame = results[0].plot()  # Kutu ve etiketleri ekler

    # Görüntüyü göster
    cv2.imshow("YOLO Webcam Tespit", annotated_frame)

    # Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()