import urllib.request
from pathlib import Path

def download_video():
    target_path = Path("data/raw_videos/demo.mp4")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
    print(f"Downloading video from {url}...")
    urllib.request.urlretrieve(url, target_path)
    print(f"Video saved to {target_path}")

if __name__ == "__main__":
    download_video()
