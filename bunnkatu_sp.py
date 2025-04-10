import subprocess
from pathlib import Path
from datetime import datetime

# --- 入力 ---
url_num = int(input("動画何本分？:"))
urls = []

# URLを入力
for i in range(url_num):
    url = input(f"{i+1}本目: ")
    urls.append(url)

# 画像を何枚切り出すか
num_frames = int(input("何枚切り出しますか？: "))

# --- パス設定 ---
yt_dlp_path = Path("C:/stream_ai/yt-dlp.exe")  # yt-dlpのパスに変更
ffmpeg_path = Path("C:/stream_ai/ffmpeg-2025-03-31-git-35c091f4b7-essentials_build/bin/ffmpeg.exe")  # ffmpegのパスに変更

# 出力先のパス設定
base_output_dir = Path("C:/stream_ai/test_movies")  # 出力先のフォルダを設定

# --- 出力先ディレクトリが存在しない場合、作成 ---
base_output_dir.mkdir(parents=True, exist_ok=True)

# --- 動画のダウンロードとフレーム切り出し ---
for url in urls:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = base_output_dir / f"frames_{timestamp}"
    
    # 出力フォルダを作成（親ディレクトリが存在することを確認）
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # ダウンロードした動画の保存場所
    downloaded_video = base_output_dir / f"test_video_{timestamp}.mp4"
    
    # yt-dlpで動画をダウンロード（音声と映像をマージ）
    print(f"🎥 {url} をダウンロード中...")
    subprocess.run([
        str(yt_dlp_path),
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "-o", str(downloaded_video),
        "--merge-output-format", "mp4",  # 動画と音声を自動でマージ
        "--ffmpeg-location", str(ffmpeg_path),  # ffmpegのパスを指定
        url
    ])

    # ダウンロード完了メッセージ
    if downloaded_video.exists():
        print(f"動画がダウンロードされました: {downloaded_video}")
    else:
        print("ダウンロードに失敗しました。")
        continue  # ダウンロード失敗した場合は次のURLに進む

    # ffmpegで動画からフレームを切り出し
    output_pattern = str(output_folder / f"output_%03d.jpg")
    command = [
        str(ffmpeg_path),
        "-i", str(downloaded_video),
        "-vf", "fps=1/5",  # 5秒ごとに切り出し
        "-vframes", str(num_frames),
        output_pattern
    ]

    print(f"{num_frames} 枚の画像を切り出し中...")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        print("画像切り出し完了！")
    else:
        print("エラーが発生しました:")
        print(result.stderr)

    # 動画の一時ファイルを削除
    if downloaded_video.exists():
        downloaded_video.unlink()
        print(f"一時ファイル（{downloaded_video}）を削除しました。")
