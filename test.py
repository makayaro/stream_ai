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
            if used[i]:
                continue
            box1 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                box2 = merged[j]
                if iou(box1, box2) > iou_threshold:
                    # マージして新しいボックスに
                    new_box = [(box1[k] + box2[k]) / 2 for k in range(4)]
                    new_merged.append(new_box)
                    used[i] = True
                    used[j] = True
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
merged_file = base.with_suffix(".mp4")  # ✅ 自動で作成される結合済みファイル
# --- 動画DL ---
print("📥 YouTube動画を高画質でダウンロード中...")
subprocess.run([
    str(yt_dlp_path),
    "--merge-output-format", "mp4",  # ← これを追加！
    "-f", "bestvideo+bestaudio",
    "-o", str(merged_file),  # 直接 .mp4 に出力
    url
])

# --- 切り出し ---
subprocess.run([
    str(ffmpeg_path), "-y", "-i", str(merged_file),
    "-t", "10", "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-c:a", "aac",
    "-f", "mp4",  # ← これを追加して出力形式を強制
    str(short_video)
])

# --- フル動画のフレームを画像として保存 ---
frames_dir = output_dir / "full_frames"
frames_dir.mkdir(exist_ok=True)

# test_video_short.mp4 → test_video_short_1.jpg, test_video_short_2.jpg, ...
subprocess.run([
    str(ffmpeg_path),
    "-i", str(short_video),
    str(frames_dir / "test_video_short_%d.jpg")
])


# --- YOLO推論 ---
predict_dir = output_dir / "predict"
print("🔎 YOLOで推論中...")
subprocess.run([
    "yolo",
    "task=detect",
    "mode=predict",
    f"model={model_path}",
    f"source={short_video}",
    "save=True",
    "save_crop=True",
    "save_txt=True",  # ← これを追加！
    "imgsz=1280",
    "conf=0.2",
    f"project={output_dir}",
    "name=predict"
])


# --- ラベル読み取り＆結合＆描画 ---
predict_dir.mkdir(exist_ok=True)  # ← これを追加！
label_dir = predict_dir / "labels"
merged_dir = predict_dir / "merged_boxes"
merged_dir.mkdir(exist_ok=True)


# --- merged_boxes 内の画像一覧から input.txt を生成 ---
input_txt = output_dir / "input.txt"
with open(input_txt, "w", encoding="utf-8") as f:
    for img_file in sorted(merged_dir.glob("*.jpg")):
        f.write(f"file '{img_file.as_posix()}'\n")  # ← これが重要


# --- input.txt を使って動画化 ---
final_video = output_dir / "tagged_video.mp4"
subprocess.run([
    str(ffmpeg_path),
    "-y",
    "-f", "concat",
    "-safe", "0",
    "-r", "30",  # フレームレート
    "-i", str(input_txt),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    str(final_video)
])
print(f"🎞️ 枠付き動画を出力しました: {final_video}")


# --- YOLO推論後のラベルをもとに、フルフレームに枠を描画 ---
for txt_file in label_dir.glob("*.txt"):
    img_file = frames_dir / txt_file.with_suffix(".jpg").name  # フルサイズ画像（例: test_video_short_1.jpg）

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

print("\n✅ 分析＆結合＆描画すべて完了！")
print(f"📁 出力先: {output_dir}")
