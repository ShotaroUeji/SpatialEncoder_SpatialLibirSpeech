import pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from requests.adapters import HTTPAdapter, Retry
import time

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/spatial-librispeech/v1/ambisonics"
OUT  = Path("/home/takamichi-lab-pc09/SpatialLibriSpeech/takamichi09/SpatialLibriSpeech")
META = "metadata.parquet"        # すでに持っているファイル

MAX_WORKERS = 4 
meta = pd.read_parquet(META)
sess = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 502, 503, 504],   # これらは自動リトライ対象
    allowed_methods=["GET"]
)
sess.mount("https://", HTTPAdapter(max_retries=retries))

def fetch(row):
    sid   = f"{row.sample_id:06d}"
    split = row.split
    url   = f"{BASE}/{sid}.flac"
    tgt   = OUT / split / f"{sid}.flac"
    if tgt.exists():
        return

    try:
        # requests.Session() での自動リトライを使うので、単純に get→write
        with sess.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            tgt.parent.mkdir(parents=True, exist_ok=True)
            with open(tgt, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
    except Exception as e:
        print(f"[{sid}] failed even after retries: {e}")

from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    list(ex.map(fetch, (row for _, row in meta.iterrows())))
