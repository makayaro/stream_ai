from pathlib import Path
import subprocess
from datetime import datetime
import numpy as np
import cv2
import torch  # YOLOv8の推論に必要

# === 入力 ===
url = input("🎥 分析したいYouTubeのURLを入力してください：\n")

# === パス設定 ===
yt_dlp_path = Path("C:/stream_ai/yt-dlp.exe")
ffmpeg_path = Path("C:/stream_ai/ffmpeg-2025-03-31-git-35c091f4b7-essentials_build/bin/ffmpeg.exe")
model_path = Path("C:/stream_ai/runs/detect/train7/weights/best.pt")  # 学習済みモデルのパス

# === 出力ディレクトリ作成 ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"C:/stream_ai/outputs/{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)

downloaded_base = output_dir / "test_video"
short_video = output_dir / "test_video_short.mp4"

# === ダウンロード ===
print("📥 YouTube動画を高画質でダウンロード中...")
subprocess.run([
    str(yt_dlp_path),
    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
    "-o", str(downloaded_base) + ".%(ext)s",
    url
])

video_file = downloaded_base.with_suffix(".f399.mp4")
audio_file = downloaded_base.with_suffix(".f140.m4a")
merged_file = downloaded_base.with_suffix(".mp4")

# === マージ処理（映像＋音声） ===
if not merged_file.exists():
    print("🎬 映像と音声を結合中...")
    subprocess.run([
        str(ffmpeg_path),
        "-y",
        "-i", str(video_file),
        "-i", str(audio_file),
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        str(merged_file)
    ])

# === マージ失敗チェック ===
if not merged_file.exists():
    print(f"❌ マージに失敗しました: {merged_file}")
    exit()

# === 3分切り出し ===
print("✂️ 3分間だけに切り出し中...")
subprocess.run([
    str(ffmpeg_path),
    "-y",
    "-i", str(merged_file),
    "-t", "180",
    "-c:v", "libx264",
    "-crf", "23",
    "-preset", "fast",
    "-c:a", "aac",
    str(short_video)
])

# === YOLOv8 推論（配信者カメラ枠を検出） ===
print("🔎 YOLOv8でカメラ枠を検出中...")
subprocess.run([
    "yolo",
    "task=detect",
    "mode=predict",
    f"model={model_path}",
    f"source={short_video}",
    "save=True",
    "save_crop=True",
    "imgsz=1280",
    "conf=0.2",
    f"project={output_dir}",
    "name=predict"
])

# === YOLOv8推論結果の読み込み ===
def load_yolo_results(output_dir):
    # 保存されているYOLOv8の結果ファイル（例: predict/labels）を取得
    label_dir = output_dir / "predict/labels"
    detected_boxes = []

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as file:
            for line in file:
                parts = line.strip().split()
                # (class_id, x_center, y_center, width, height) の順番
                detected_boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

    return detected_boxes

# YOLOv8の結果をロード
detected_boxes = load_yolo_results(output_dir)

# === 枠の統合処理 ===
def merge_bboxes(bboxes, iou_threshold=0.5):
    """
    複数のバウンディングボックスを統合する関数
    近接しているボックスを統合します。
    """
    if len(bboxes) == 0:
        return []

    def iou(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        inter_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / union_area

    merged_bboxes = []
    for bbox in bboxes:
        if not merged_bboxes:
            merged_bboxes.append(bbox)
            continue

        merged = False
        for i, merged_bbox in enumerate(merged_bboxes):
            if iou(bbox, merged_bbox) > iou_threshold:
                merged_bboxes[i] = [
                    (bbox[0] + merged_bbox[0]) / 2, 
                    (bbox[1] + merged_bbox[1]) / 2,
                    (bbox[2] + merged_bbox[2]) / 2,
                    (bbox[3] + merged_bbox[3]) / 2
                ]
                merged = True
                break

        if not merged:
            merged_bboxes.append(bbox)

    return merged_bboxes

# ボックスを統合
merged_boxes = merge_bboxes(detected_boxes)

# 統合された枠を新しい動画に描画する処理
cap = cv2.VideoCapture(str(short_video))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(str(output_dir / "final_output.mp4"), fourcc, 30, (640, 360))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 統合されたボックスを描画
    for box in merged_boxes:
        x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * frame.shape[1])
        y1 = int((y_center - height / 2) * frame.shape[0])
        x2 = int((x_center + width / 2) * frame.shape[1])
        y2 = int((y_center + height / 2) * frame.shape[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out_video.write(frame)

cap.release()
out_video.release()

print(f"✅ 分析完了！最終的な動画は {output_dir / 'final_output.mp4'} に保存されました。")
