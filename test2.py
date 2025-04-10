from pathlib import Path
import subprocess
from datetime import datetime
import numpy as np # type: ignore
import cv2 # type: ignore

def merge_bboxes(bboxes, iou_threshold=0.05):#iou_threshold枠がどんだけかぶってるか
    """
    複数のバウンディングボックスを統合する関数
    近接しているボックスを統合します。
    iou_threshold: 統合するためのIoU閾値
    """
    if len(bboxes) == 0:
        return []

    # IoU計算関数
    def iou(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        inter_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        union_area = w1 * h1 + w2 * h2 - inter_area
        iou_value = inter_area / union_area

        # IoUの値を表示してデバッグする
        print(f"IoU between {bbox1} and {bbox2}: {iou_value}")
    
        return iou_value  

    # 重なっているボックスを統合
    merged_bboxes = []
    for bbox in bboxes:
        if not merged_bboxes:
            merged_bboxes.append(bbox)
            continue

        merged = False
        for i, merged_bbox in enumerate(merged_bboxes):
            if iou(bbox, merged_bbox) > iou_threshold:
                # 統合する場合、平均を取って一つの枠にする
                merged_bboxes[i] = [
                    (bbox[0] + merged_bbox[0]) / 2,  # x_center
                    (bbox[1] + merged_bbox[1]) / 2,  # y_center
                    (bbox[2] + merged_bbox[2]) / 2,  # width
                    (bbox[3] + merged_bbox[3]) / 2   # height
                ]
                merged = True
                break

        if not merged:
            merged_bboxes.append(bbox)

    return merged_bboxes

def merge_audio_video(video_file, audio_file, output_file, ffmpeg_path):
    """
    映像と音声を統合する関数
    video_file: 映像ファイル（mp4）
    audio_file: 音声ファイル（m4a）
    output_file: 出力先の動画ファイル
    ffmpeg_path: ffmpegのパス
    """
    print("🎬 映像と音声を結合中...")
    subprocess.run([
        str(ffmpeg_path),
        "-y",  # 出力先があれば上書き
        "-i", str(video_file),  # 映像ファイル
        "-i", str(audio_file),  # 音声ファイル
        "-c:v", "copy",         # 映像は再エンコードしない
        "-c:a", "aac",          # 音声はaacにエンコード
        "-strict", "experimental",  # AACの使用
        str(output_file)        # 出力ファイル
    ])

    if not output_file.exists():
        print(f"❌ マージに失敗しました: {output_file}")
        exit()
    print("🎬 音声と映像の結合完了！")

# === 入力 ===
url = input("🎥 分析したいYouTubeのURLを入力してください：\n")

# === パス設定 ===
yt_dlp_path = Path("C:/stream_ai/yt-dlp.exe")
ffmpeg_path = Path("C:/stream_ai/ffmpeg-2025-03-31-git-35c091f4b7-essentials_build/bin/ffmpeg.exe")
model_path = Path("C:/stream_ai/runs/detect/train7/weights/best.pt")

# === 出力ディレクトリ作成 ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"C:/stream_ai/outputs/{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)

downloaded_base = output_dir / "test_video"
short_video = output_dir / "test_video_short.mp4"

# === ダウンロード ===
print("📥 YouTube動画を高画質でダウンロード中...")
subprocess.run([  # 動画のダウンロード
    str(yt_dlp_path),
    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
    "-o", str(downloaded_base) + ".%(ext)s",
    url
])

video_file = downloaded_base.with_suffix(".f399.mp4")
audio_file = downloaded_base.with_suffix(".f140.m4a")
merged_file = downloaded_base.with_suffix(".mp4")

# === マージ処理（映像＋音声） ===
merge_audio_video(video_file, audio_file, merged_file, ffmpeg_path)

# === 3分切り出し ===
print("切り出し中...")
subprocess.run([
    str(ffmpeg_path),
    "-y",
    "-i", str(merged_file),
    "-t", "10",
    "-c:v", "libx264",
    "-crf", "23",
    "-preset", "fast",
    "-c:a", "aac",
    str(short_video)
])

# === 推論（YOLO） ===
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

# === 枠の統合処理 ===
detected_boxes = [...]  # YOLOv8の検出結果
merged_boxes = merge_bboxes(detected_boxes)

# 統合された枠を新しい動画に描画する処理を追加
for box in merged_boxes:
    pass  # 枠を描画する処理（例えば、OpenCVなどを使って



# YOLO出力結果（画像＆ラベル）を読み込んで処理
predict_dir = output_dir / "predict"  # YOLO推論後の画像＆txtがある場所
merged_dir = predict_dir / "merged_boxes"
merged_dir.mkdir(exist_ok=True)

for txt_file in predict_dir.glob("*.txt"):
    img_file = txt_file.with_suffix(".jpg")
    if not img_file.exists():
        continue

    # --- YOLO結果の読み込み ---
    bboxes = []
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x, y, w, h = map(float, parts[1:5])
                bboxes.append([x, y, w, h])

    # --- 枠を統合 ---
    merged = merge_bboxes(bboxes, iou_threshold=0.2)

    # --- 画像読み込み & 描画 ---
    img = cv2.imread(str(img_file))
    h_img, w_img = img.shape[:2]

    for box in merged:
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w_img)
        y1 = int((cy - bh / 2) * h_img)
        x2 = int((cx + bw / 2) * w_img)
        y2 = int((cy + bh / 2) * h_img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- 保存 ---
    out_path = merged_dir / img_file.name
    cv2.imwrite(str(out_path), img)

print("✅ 統合された枠を描画して保存しました！")
print(f"📁 出力先: {merged_dir}")

print(f"\n✅ 分析完了！\n📁 保存先: {output_dir}")
