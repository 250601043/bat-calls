import os
import zipfile

import pandas as pd
import requests

API = "https://api.figshare.com/v2"
CONTEXTS = [2, 3, 4, 9, 12]


def download(name):
    if not os.path.exists(f"data/{name}"):
        with requests.get(urls[name], stream=True, timeout=60) as response, open("data/part", "wb") as f:
            response.raise_for_status()
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)
        os.rename("data/part", f"data/{name}")


urls = {}
for article in requests.get(f"{API}/collections/3666502/articles", params={"page_size": 100}).json():
    for file in requests.get(f"{API}/articles/{article['id']}/files").json():
        urls[file["name"]] = file["download_url"]

os.makedirs("data/audio", exist_ok=True)
download("Annotations.csv")
download("FileInfo.csv")
annotations = pd.read_csv("data/Annotations.csv")
info = pd.read_csv("data/FileInfo.csv", usecols=range(6))
needed = info[info.FileID.isin(annotations[annotations.Context.isin(CONTEXTS)].FileID)]
for folder, names in needed.groupby("File folder")["File name"]:
    missing = [name for name in names if not os.path.exists(f"data/audio/{name}")]
    if missing:
        download(f"{folder}.zip")
        with zipfile.ZipFile(f"data/{folder}.zip") as archive:
            for name in missing:
                archive.extract(name, "data/audio")
        os.remove(f"data/{folder}.zip")
    print(folder, len(missing), flush=True)
