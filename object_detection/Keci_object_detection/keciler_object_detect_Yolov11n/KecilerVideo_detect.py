import cv2 #opencv kütüphanesi
import time #zaman ölçmek için kullanılır
import random #random renkler oluşturmak için kullanılır
import argparse #komut satırı argümanlarını yönetmek için kullanılır
import numpy as np

def loadSource(source_file):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    key = 1 # Görüntü modunda tek bir kare yakalamak için kullanılır.
    frame = None
    cap = None
    
    #kaynağın görüntü mü video mu olduğunu belirlemek.
    if(source_file == "0"):
        # image_type: Kaynağın bir görüntü mü (True) yoksa video mu (False) olduğunu belirtir.
        image_type = False
        source_file = 0    
    else:
        image_type = source_file.split('.')[-1].lower() in img_formats

    #eğer kaynak görüntü ise cv2.imread çalışır
    if(image_type):
        frame = cv2.imread(source_file)
        key = 0
    #eğer kaynak video ise cv2.VideoCapture çalışır
    else:
        cap = cv2.VideoCapture(source_file)

    return image_type, key, frame, cap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #--source: Görüntü ya da video kaynağının yolu.
    parser.add_argument("--source", type=str, default="data/videos/Keçiler.mp4", help="Video")
    #--names: Tespit edilen sınıfların isimlerinin bulunduğu dosyanın yolu.
    parser.add_argument("--names", type=str, default="data/class.names", help="Object Names")
    #--model: YOLOv8 model dosyasının yolu (ONNX formatında).
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Pretrained Model")
    #--tresh: Minimum güven (confidence) eşiği.
    parser.add_argument("--tresh", type=float, default=0.25, help="Confidence Threshold")
    #--thickness: Çizilecek sınırlayıcı kutuların kalınlığı.
    parser.add_argument("--thickness", type=int, default=2, help="Line Thickness on Bounding Boxes")
    args = parser.parse_args()    
    
    #cv2.dnn.readNet: OpenCV'nin DNN modülü ile ONNX formatındaki önceden 
    #eğitilmiş modeli yükler.
    model = cv2.dnn.readNet(args.model)

    IMAGE_SIZE = 640
    NAMES = []
    #Tespit edilecek nesnelerin sınıf isimlerini dosyadan okur ve bir listeye kaydeder.
    with open(args.names, "r") as f:
        NAMES = [cname.strip() for cname in f.readlines()] #f.readlines(): Dosyadaki her bir satırı okur ve bir liste döndürür.
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in NAMES]

    source_file = args.source    
    image_type, key, frame, cap = loadSource(source_file)
    grabbed = True
    # Video kaynaklarında her bir karenin başarıyla okunduğunu kontrol 
    # etmek için bir başlangıç durumu ayarlanır.

    while(1):
        # Eğer kaynak bir video veya kamera ise, bu bloğa girilir ve
        # video/kamera kareleri okunur.
        # Görüntü kaynakları için yalnızca tek bir kare alınır.
        if not image_type:
            (grabbed, frame) = cap.read()
            
        # kaynaktan artık yeni bir kare okunamadığını gösterir. Bunun birkaç nedeni olabilir:
        # Video dosyası bitmiştir. Kamera bağlantısı kesilmiştir. Kaynaktan veri alınamıyordur.
        # Eğer grabbed değeri False ise, bu kod çalışır.
        if not grabbed:
            exit()

        image = frame.copy()#orijinal görüntü korunur ve işlenecek 
                            # görüntü (image) ayrı bir değişkende tutulur.

        # Görüntüyü bir derin öğrenme modeline uygun formatta hazırlamak.
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (IMAGE_SIZE, IMAGE_SIZE), swapRB=True, crop=False)
        # Algılanan nesnelerin sınıf kimliklerini (class_ids), 
        # güven skorlarını (confs) ve koordinatlarını (boxes) 
        # saklayacak boş listeler oluşturmak.
        class_ids, confs, boxes = list(), list(), list()

        model.setInput(blob) #Model, bu blob'u kullanarak tahminler yapacak.
        preds = model.forward() #Modelden çıkarımları almak.
        preds = preds.transpose((0, 2, 1)) #Çıkışın eksenlerini yeniden düzenlemek.

        # Giriş görüntüsünün yüksekliğini (image_height) ve genişliğini (image_width) almak.
        image_height, image_width, _ = image.shape
        x_factor = image_width / IMAGE_SIZE #Genişlik ölçekleme faktörü.
        y_factor = image_height / IMAGE_SIZE #Yükseklik ölçekleme faktörü.

        rows = preds[0].shape[0] #Modelin çıkışındaki satır sayısını (tahmin edilen nesne sayısı) belirlemek.

        for i in range(rows):
            row = preds[0][i] #Tahmin edilen her satır üzerinde işlem yapmak.
            conf = row[4] #Her tahminin genel güven skorunu almak.
            
            # Tahmin edilen sınıf skorlarını (her sınıf için olasılık değerleri) almak.
            # İlk dört değer genelde koordinat bilgilerini temsil eder.
            classes_score = row[4:]
            # cv2.minMaxLoc: Bir dizideki minimum ve maksimum değerlerin konumlarını bulur.
            _,_,_, max_idx = cv2.minMaxLoc(classes_score)
            # max_idx[1]: En yüksek skorun sınıf kimliği.
            class_id = max_idx[1]
            # Amaç: Sınıf skorunun bir eşik değerinden (örneğin, 0.25) büyük olup olmadığını kontrol etmek.
            # Detaylar: Bu, yalnızca yeterince güvenilir tahminlerin işleme alınmasını sağlar.
            if (classes_score[class_id] > .25):
                confs.append(classes_score[class_id])#confs: Tahmin edilen sınıfın güven skorlarını içerir.
                label = NAMES[int(class_id)]#NAMES: Sınıf isimlerinin bulunduğu bir liste (örneğin, ["cat", "dog"]).
                class_ids.append(class_id)#class_ids: Tahmin edilen sınıf kimliklerini tutar.
                
                # Tahmin edilen kutu koordinatlarını gerçek görüntü boyutlarına 
                # ölçeklendirmek ve kutuları depolamak.
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)                
        # Birbirine çok benzeyen (üst üste binen) tahminlerden en güvenilir olanını seçmek.
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.2, 0.5)         
        
        for i in indexes:
            # Tahmin edilen kutu, sınıf kimliği ve güven skorunu almak.
            box = boxes[i]
            class_id = class_ids[i]
            score = confs[i]
            
            # Kutunun başlangıç koordinatlarını (left, top) ve boyutlarını (width, height) ayırmak
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            # Görüntüde algılanan nesnenin etr çizmeafına kutu (dikdörtgen)k.
            cv2.rectangle(image, (left, top), (left + width, top + height), COLORS[class_id], args.thickness)
            
            # Nesnenin adını (name) ve güven skorunu (score) birleştirerek etiket oluşturmak.
            name = NAMES[class_id]    
            score = round(float(score), 1)            
            name = name +" %"+ f'{str(score*100)}'
            
            print("score: ",score)
            print("name: ",name)

            # Etiketin yazı boyutunu (font_size) ve kutudan uzaklığını (margin) ayarlamak.
            font_size = args.thickness / 2.5
            margin = args.thickness * 2 
            #args.thickness: Kutunun kalınlığına bağlı olarak yazının boyutu ve konumu belirlenir.
            
            #  Nesne adı ve güven skorunu kutunun üzerine yazmak.
            cv2.putText(image, name, (left, top - margin), cv2.FONT_HERSHEY_SIMPLEX, font_size, COLORS[class_id], args.thickness)
            # (left, top - margin): Yazının başlangıç noktası.
            # cv2.FONT_HERSHEY_SIMPLEX: Yazı tipi.
            # font_size: Yazı boyutu.
            # COLORS[class_id]: Sınıfa özgü renk.
            # args.thickness: Yazı çizgi kalınlığı.

        grabbed = False
        cv2.imshow("Detected",image)#Güncellenmiş görüntüyü bir pencerede göstermek.
        # "Detected": Pencerenin adı.
        # image: Üzerine kutular ve yazılar eklenmiş görüntü.

        if cv2.waitKey(key) ==  ord('q'):
            break        
        # cv2.waitKey(key): Belirtilen süre boyunca klavye girişini bekler (key, milisaniye cinsinden).
        # ord('q'): Kullanıcı 'q' tuşuna basarsa, döngü sonlanır.
        
        
        
        
        