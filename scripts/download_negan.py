import os
import sys
from pathlib import Path
import requests
import zipfile

BASE = Path(__file__).resolve().parents[1]
fonts_dir = BASE / 'ui' / 'public' / 'fonts'
fonts_dir.mkdir(parents=True, exist_ok=True)

zip_url = 'https://www.allfreefonts.co/wp-content/uploads/download-manager-files/Negan.zip'
zip_path = fonts_dir / 'Negan.zip'

print('Downloading', zip_url)
resp = requests.get(zip_url, stream=True, timeout=30)
resp.raise_for_status()
with open(zip_path, 'wb') as f:
    for chunk in resp.iter_content(1024*8):
        if chunk:
            f.write(chunk)

print('Downloaded to', zip_path)
try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(fonts_dir)
    print('Extracted to', fonts_dir)
except zipfile.BadZipFile:
    print('Downloaded file is not a valid zip')

print('\nFiles:')
for p in sorted(fonts_dir.iterdir()):
    print(p.name)

print('\nDone')
