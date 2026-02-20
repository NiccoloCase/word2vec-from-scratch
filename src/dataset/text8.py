import os
import urllib.request
import zipfile
from config import DATA_DIR

TEXT8_URL  = "http://mattmahoney.net/dc/text8.zip"
TEXT8_ZIP  = "text8.zip"
TEXT8_FILE = "text8"

def download_text8() -> str:
    dest = DATA_DIR
    path = os.path.join(dest, TEXT8_FILE)
    
    if os.path.exists(path):
        print(f"Dataset already present at {path}")
        return path
    
    zip_path = os.path.join(dest, TEXT8_ZIP)
    
    # extract the zip file to the destination directory
    urllib.request.urlretrieve(TEXT8_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    # delete the zip file after extraction
    os.remove(zip_path)

    return path


def load_text8(path: str, max_tokens: int | None = None) -> list[str]:
    with open(path, "r", encoding="utf-8") as f: tokens = f.read().split()
    if max_tokens: tokens = tokens[:max_tokens]
    return tokens

