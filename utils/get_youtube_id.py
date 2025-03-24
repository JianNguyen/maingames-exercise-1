import re

def get_youtube_id(url):
    pattern = r"(?:youtu\.be/|youtube\.com/(?:.*v=|embed/|v/|.+?/))([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Example Usage:
print(get_youtube_id("https://www.youtube.com/watch?v=YpjwHazhknA"))  # Output: abc123XYZ90
print(get_youtube_id("https://youtu.be/abc123XYZ90"))  # Output: abc123XYZ90