from pathlib import Path
import subprocess
from datetime import datetime
import numpy as np # type: ignore
import cv2 # type: ignore

# --- 関数 ---
def yolo_to_abs(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2

# --- 設定 ---
output_dir = Path("C:/stream_ai/outputs/20250403_202623")  # ←適宜変更
label_dir = output_dir / "predict" / "labels"
frames_dir = output_dir / "full_frames"
cropped_dir = output_dir / "cropped_box_frames"
boxed_dir = output_dir / "fixed_box_frames"
ffmpeg_path = Path("C:/stream_ai/ffmpeg-2025-03-31-git-35c091f4b7-essentials_build/bin/ffmpeg.exe")

boxed_dir.mkdir(exist_ok=True)
cropped_dir.mkdir(exist_ok=True)

# --- 最大の枠を探す ---
all_boxes_abs = []
img_w, img_h = 1280, 720  # 解像度（YOLOv8推論時に使った画像サイズ）

for txt_file in sorted(label_dir.glob("*.txt")):
    with open(txt_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, cx, cy, w, h = map(float, parts[:5])
                x1, y1, x2, y2 = yolo_to_abs(cx, cy, w, h, img_w, img_h)
                all_boxes_abs.append((x1, y1, x2, y2))

if not all_boxes_abs:
    print("❌ 検出枠が見つかりません。YOLOの結果を確認してください。")
    exit()

# 最大のバウンディングボックス（全枠の最大カバー）
x1s, y1s, x2s, y2s = zip(*all_boxes_abs)
fx1, fy1, fx2, fy2 = min(x1s), min(y1s), max(x2s), max(y2s)
print(f"📦 固定枠: ({fx1}, {fy1}), ({fx2}, {fy2})")

# --- 枠を描画＆切り抜き ---
for frame_file in sorted(frames_dir.glob("*.jpg")):
    img = cv2.imread(str(frame_file))
    if img is None:
        continue

    # 枠を描画
    boxed = img.copy()
    cv2.rectangle(boxed, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
    cv2.imwrite(str(boxed_dir / frame_file.name), boxed)

    # 切り抜き（2の倍数に）
    cropped = img[fy1:fy2, fx1:fx2]
    ch, cw = cropped.shape[:2]
    ch -= ch % 2
    cw -= cw % 2
    cropped = cv2.resize(cropped, (cw, ch))
    cv2.imwrite(str(cropped_dir / frame_file.name), cropped)

# --- 動画出力 ---
fixed_video = output_dir / "fixed_frame_boxed.mp4"
cropped_video = output_dir / "cropped_box_only.mp4"

subprocess.run([
    str(ffmpeg_path), "-y", "-framerate", "30",
    "-i", str(boxed_dir / "test_video_short_%d.jpg"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    str(fixed_video)
])

subprocess.run([
    str(ffmpeg_path), "-y", "-framerate", "30",
    "-i", str(cropped_dir / "test_video_short_%d.jpg"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    str(cropped_video)
])

print(f"🎞️ 固定枠付き動画: {fixed_video}")
print(f"✂️ 切り抜き動画: {cropped_video}")
