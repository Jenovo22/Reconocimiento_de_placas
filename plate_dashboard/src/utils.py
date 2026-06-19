from pathlib import Path


def list_videos(folder: Path):
    return [f for f in folder.glob("*") if f.suffix in [".mp4", ".avi", ".mov"]]