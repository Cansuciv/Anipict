# Kütüphane Yüklemeleri
# pip install onnx onnxruntime onnxsim
# Videolar üzerinden tespit işlemi yapmak için keciler_best.pt dosyası onnx formatına dönüştürüldü

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("keciler_best.pt")

# Export the model to ONNX format
model.export(format="onnx")  # creates 'yolo11n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("keciler_best.onnx")
