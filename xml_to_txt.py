import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

# === パス設定 ===
input_dir = Path("C:/stream_ai/test_movies")  # 各 frames_〜 フォルダが入っている場所
output_dir = Path("C:/stream_ai/dataset")     # YOLO学習用の出力先

# === 出力先初期化 ===
if output_dir.exists():
    shutil.rmtree(output_dir)
for split in ["train", "val"]:
    (output_dir / f"images/{split}").mkdir(parents=True)
    (output_dir / f"labels/{split}").mkdir(parents=True)

# === XML → YOLO変換関数 ===
def convert_xml_to_yolo_format(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_dict = {"stream_tag": 0}  # 必要に応じて追加
        class_id = class_dict.get(class_name, -1)
        if class_id == -1:
            continue
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return "\n".join(yolo_lines)

# === 分割対象を収集 ===
samples = []
for frame_folder in input_dir.glob("frames_*"):
    for img_file in frame_folder.glob("*.jpg"):
        xml_file = img_file.with_suffix(".xml")
        if not xml_file.exists():
            continue
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if not root.findall("object"):
            continue
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)
        yolo_txt = convert_xml_to_yolo_format(xml_file, img_width, img_height)
        if not yolo_txt.strip():
            continue
        samples.append((img_file, yolo_txt))

# === ランダムに train/val 分割 ===
random.shuffle(samples)
split_index = int(len(samples) * 0.8)
train_samples = samples[:split_index]
val_samples = samples[split_index:]

for split, split_samples in zip(["train", "val"], [train_samples, val_samples]):
    for img_file, yolo_txt in split_samples:
        new_name = img_file.parent.name + "_" + img_file.name
        shutil.copy(img_file, output_dir / f"images/{split}" / new_name)
        label_path = output_dir / f"labels/{split}" / new_name.replace(".jpg", ".txt")
        with open(label_path, "w") as f:
            f.write(yolo_txt)

print("✅ 完了！YOLOv8形式に変換して train/val に分割しました。")
print(f"📁 images/train: {output_dir / 'images/train'}")
print(f"📁 images/val  : {output_dir / 'images/val'}")