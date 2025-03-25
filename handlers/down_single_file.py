import yt_dlp as youtube_dl
import json
from tqdm import tqdm

ydl_opts = {
    'quiet': True,
    'no_warnings': True,
    'verbose': False,
    'outtmpl': '../downloads/%(title)s.%(ext)s',
    'format': 'bestvideo/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp4',
        # 'preferredquality': '192'
    }],
    'postprocessor_args': [
        '-ar', '44100', '-ac', '1', '-ab', '128k'
    ],
    'prefer_ffmpeg': True,
    'keepvideo': False,
            }

url = "https://www.youtube.com/watch?v=Tefu_NvcC0k"
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])