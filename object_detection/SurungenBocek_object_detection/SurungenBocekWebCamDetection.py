from ultralytics import YOLO
import cv2

model = YOLO("SurungenBocek_best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Düşük conf threshold ve tüm sınıfları algıla
    results = model.predict(frame, imgsz=640, conf=0.3)
    
    # Algılanan sınıfları konsola yazdır (debug için)
    if len(results[0]) > 0:
        print("Algılanan sınıflar:", results[0].boxes.cls)
    
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Webcam", annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()