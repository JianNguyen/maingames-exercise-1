import yt_dlp as youtube_dl
import os
#
# ydl_opts = {
#         'quiet': True,
#         'no_warnings': True,
#         'verbose': False,
#         'outtmpl': './%(title)s.%(ext)s',
#         'format': 'bestaudio/best',
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'wav',
#         }],
#         'postprocessor_args': [
#             '-ar', '44100', '-ac', '1', '-ab', '128k'
#         ],
#         'prefer_ffmpeg': True,
#         'keepvideo': False,
#                 }

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


# def download_youtube_complete(url, output_path="./"):
#     """Download complete video with audio in MP4 format"""
#     ydl_opts = {
#         'quiet': True,
#         'no_warnings': True,
#         'verbose': False,
#         'outtmpl': os.path.join(output_path, '%(title)s_complete.%(ext)s'),
#         'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',  # MP4 video with audio
#         'merge_output_format': 'mp4',
#         'prefer_ffmpeg': True,
#     }
#
#     with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(url, download=True)
#         return ydl.prepare_filename(info)

def download_all_formats(url, output_dir="./"):
    """Download a YouTube video in all three formats"""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download in all formats
    audio_path = download_youtube_audio(url, output_dir)
    video_only_path = download_youtube_video_only(url, output_dir)
    # complete_path = download_youtube_complete(url, output_dir)

    return {
        "audio": audio_path,
        "video_only": video_only_path,
        # "complete": complete_path
    }

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=Tefu_NvcC0k"
    files = download_all_formats(url, "../downloads")
    print(files)

# url = "https://www.youtube.com/watch?v=Tefu_NvcC0k"
