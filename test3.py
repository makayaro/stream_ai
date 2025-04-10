from pathlib import Path
import subprocess
from datetime import datetime
import numpy as np  # type: ignore
import cv2  # type: ignore

def merge_bboxes(bboxes, iou_threshold=0.2):
    def iou(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        inter_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / union_area if union_area else 0

    merged = bboxes[:]
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]: continue
            box1 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]: continue
                box2 = merged[j]
                if iou(box1, box2) > iou_threshold:
                    new_box = [(box1[k] + box2[k]) / 2 for k in range(4)]
                    new_merged.append(new_box)
                    used[i] = used[j] = True
                    changed = True
                    break
            if not used[i]:
                new_merged.append(box1)
                used[i] = True
        merged = new_merged
    return merged

# --- 入力 ---
url = input("🎥 分析したいYouTubeのURLを入力してください：\n")

# --- パス設定 ---
yt_dlp_path = Path("C:/stream_ai/yt-dlp.exe")
ffmpeg_path = Path("C:/stream_ai/ffmpeg-2025-03-31-git-35c091f4b7-essentials_build/bin/ffmpeg.exe")
model_path = Path("C:/stream_ai/runs/detect/train7/weights/best.pt")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"C:/stream_ai/outputs/{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)

base = output_dir / "test_video"
short_video = output_dir / "test_video_short.mp4"
merged_file = base.with_suffix(".mp4")

# --- YouTube動画ダウンロード ---
print("📥 YouTube動画をダウンロード中...")
subprocess.run([
    str(yt_dlp_path),
    "--merge-output-format", "mp4",
    "-f", "bestvideo+bestaudio",
    "-o", str(merged_file),
    url
])

# --- 動画切り出し ---
subprocess.run([
    str(ffmpeg_path), "-y", "-i", str(merged_file),
    "-t", "10", "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-c:a", "aac",
    str(short_video)
])

# --- 動画からフレームを保存 ---
frames_dir = output_dir / "full_frames"
frames_dir.mkdir(exist_ok=True)
subprocess.run([
    str(ffmpeg_path), "-i", str(short_video),
    str(frames_dir / "test_video_short_%d.jpg")
])

# --- YOLO推論 ---
predict_dir = output_dir / "predict"
print("🔎 YOLOで推論中...")
subprocess.run([
    "yolo", "task=detect", "mode=predict",
    f"model={model_path}", f"source={short_video}",
    "save=True", "save_crop=True", "save_txt=True",
    "imgsz=1280", "conf=0.2",
    f"project={output_dir}", "name=predict"
])

# --- ラベル読み取り＆描画 ---
label_dir = predict_dir / "labels"
merged_dir = predict_dir / "merged_boxes"
merged_dir.mkdir(exist_ok=True)

for txt_file in sorted(label_dir.glob("*.txt")):
    img_file = frames_dir / txt_file.with_suffix(".jpg").name
    if not img_file.exists():
        print(f"⚠️ 対応画像が存在しません: {img_file}")
        continue

    bboxes = []
    with open(txt_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                bboxes.append(list(map(float, parts[1:5])))

    if not bboxes:
        print(f"⚠️ ボックスが空です: {txt_file}")
        continue

    merged = merge_bboxes(bboxes)
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"❌ 画像が読み込めません: {img_file}")
        continue

    h, w = img.shape[:2]
    for cx, cy, bw, bh in merged:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    out_path = merged_dir / img_file.name
    cv2.imwrite(str(out_path), img)
# --- 最大の枠を計算 ---
all_boxes = []

for txt_file in label_dir.glob("*.txt"):
    with open(txt_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                all_boxes.append(list(map(float, parts[1:5])))

if not all_boxes:
    print("❌ 枠が見つかりませんでした。")
    exit()

# YOLO形式（cx, cy, w, h）→ 画像サイズをもとにピクセル座標に変換
h, w = 720, 1280  # 解像度（推定で固定）※必要に応じて変更
abs_boxes = []
for cx, cy, bw, bh in all_boxes:
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    abs_boxes.append((x1, y1, x2, y2))

# 最大範囲を計算
x1s, y1s, x2s, y2s = zip(*abs_boxes)
max_box = (min(x1s), min(y1s), max(x2s), max(y2s))

print(f"📏 最大の固定枠: {max_box}")

# --- フルフレーム画像を読み込み、枠を描画＆切り抜き ---
fixed_box_dir = output_dir / "fixed_box_frames"
cropped_box_dir = output_dir / "cropped_box_frames"
fixed_box_dir.mkdir(exist_ok=True)
cropped_box_dir.mkdir(exist_ok=True)

for frame_img in sorted(frames_dir.glob("*.jpg")):
    img = cv2.imread(str(frame_img))
    if img is None:
        continue

    # 枠を描画
    x1, y1, x2, y2 = max_box
    boxed_img = img.copy()
    cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 枠つき画像保存
    boxed_path = fixed_box_dir / frame_img.name
    cv2.imwrite(str(boxed_path), boxed_img)

    # 切り抜き画像保存（サイズを2の倍数に調整）
    cropped = img[y1:y2, x1:x2]
    h, w = cropped.shape[:2]
    new_w = w if w % 2 == 0 else w - 1
    new_h = h if h % 2 == 0 else h - 1
    cropped = cv2.resize(cropped, (new_w, new_h))  # ← ここで調整！

    cropped_path = cropped_box_dir / frame_img.name
    cv2.imwrite(str(cropped_path), cropped)


# --- 動画化（固定枠付き） ---
final_fixed_video = output_dir / "fixed_frame_boxed.mp4"
subprocess.run([
    str(ffmpeg_path),
    "-y",
    "-framerate", "30",
    "-i", str(fixed_box_dir / "test_video_short_%d.jpg"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    str(final_fixed_video)
])

# --- 動画化（切り抜き） ---
final_cropped_video = output_dir / "cropped_box_only.mp4"
subprocess.run([
    str(ffmpeg_path),
    "-y",
    "-framerate", "30",
    "-i", str(cropped_box_dir / "test_video_short_%d.jpg"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    str(final_cropped_video)
])

print(f"🎞️ 固定枠付き動画を出力しました: {final_fixed_video}")
print(f"✂️ 切り抜き動画を出力しました: {final_cropped_video}")

# --- input.txt を生成 ---
input_txt = output_dir / "input.txt"
with open(input_txt, "w", encoding="utf-8") as f:
    for img_path in sorted(merged_dir.glob("*.jpg")):
        f.write(f"file '{img_path.as_posix()}'\n")

# --- input.txt を使って動画生成 ---
final_video = output_dir / "tagged_video.mp4"
subprocess.run([
    str(ffmpeg_path),
    "-y", "-f", "concat", "-safe", "0",
    "-r", "30",
    "-i", str(input_txt),
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    str(final_video)
])

print("\n✅ すべて完了しました！")
print(f"📁 出力フォルダ: {output_dir}")
print(f"🎞️ 最終動画: {final_video}")
