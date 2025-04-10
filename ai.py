from ultralytics import YOLO

# --- 学習用設定 ---
model_path = "yolov8n.pt"  # 軽量モデル。他に yolov8s.pt や yolov8m.pt などもOK
data_yaml = "C:/stream_ai/data.yaml"  # ← data.yaml のパスを指定
epochs = 30
imgsz = 640

# --- 学習開始 ---
model = YOLO(model_path)
model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)

print("✅ 学習完了！")