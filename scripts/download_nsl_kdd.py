"""
Script to download NSL-KDD dataset from GitHub into the data/ folder.
Source: https://github.com/thinline72/nsl-kdd/tree/master/NSL_KDD_Dataset
"""

import urllib.request
from pathlib import Path

# Configuration
REPO_BASE = "https://raw.githubusercontent.com/thinline72/nsl-kdd/master/NSL_KDD_Dataset"
DATASET_FILES = [
    "KDDTest+.txt",
    "KDDTrain+.txt",
]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL to destination path."""
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"  -> Saved to {dest_path} ({dest_path.stat().st_size:,} bytes)")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main() -> None:
    # Tạo folder data nếu chưa có để tránh lỗi khi ghi file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NSL-KDD dataset to {DATA_DIR}\n")

    success_count = 0
    for filename in DATASET_FILES:
        url = f"{REPO_BASE}/{filename}"
        dest_path = DATA_DIR / filename
        if download_file(url, dest_path):
            success_count += 1
        print()

    print(f"Done: {success_count}/{len(DATASET_FILES)} files downloaded.")


if __name__ == "__main__":
    main()
