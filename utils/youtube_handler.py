import yt_dlp as youtube_dl
import os
import re


def download_youtube_audio(url, output_path="./"):
    """Download only the audio from a YouTube video in WAV format"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'verbose': False,
        'outtmpl': os.path.join(output_path, '%(title)s_audio.%(ext)s'),
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'postprocessor_args': [
            '-ar', '44100', '-ac', '1', '-ab', '128k'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        return ydl_opts["outtmpl"]


def download_youtube_video_only(url, output_path="./"):
    """Download only the video stream (no audio) from a YouTube video"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'verbose': False,
        'outtmpl': os.path.join(output_path, '%(title)s_video_only.%(ext)s'),
        'format': 'bv*[vcodec^=avc1]',
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',  # Convert to MP4 if needed
        }],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        return ydl_opts["outtmpl"]


def download_youtube_complete(url, output_path="./"):
    """Download complete video with audio in MP4 format"""
    _id = get_youtube_id(url)
    clean_url = f"https://www.youtube.com/watch?v={_id}"
    ydl_opts = {
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),  # Save file with video title
        'format': 'bestvideo+bestaudio/best',  # Best quality
        'merge_output_format': 'mp4',  # Ensure MP4 format

    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(clean_url, download=True)
        filename = ydl.prepare_filename(info)
        return filename


def download_all_formats(url, output_dir="./"):
    """Download a YouTube video in all three formats"""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download in all formats
    # audio_path = download_youtube_audio(url, output_dir)
    # video_only_path = download_youtube_video_only(url, output_dir)
    complete_path = download_youtube_complete(url, output_dir)

    return {
        # "audio": audio_path,
        # "video_only": video_only_path,
        "complete": complete_path
    }

def get_youtube_id(url):
    pattern = r"(?:youtu\.be/|youtube\.com/(?:.*v=|embed/|v/|.+?/))([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=op1QYwRAQpI&list=RDkjYW63CVbsE&index=16"
    _id = get_youtube_id(url)
    clean_url = f"https://www.youtube.com/watch?v={_id}"
    files = download_youtube_complete(clean_url, "../downloads")
    print(files)

# url = "https://www.youtube.com/watch?v=Tefu_NvcC0k"
