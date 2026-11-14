from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

CONTEXTS = {2: 0, 3: 1, 4: 2, 9: 3, 12: 4}
SAMPLES = 2 * 250000
COPIES = {0: 4, 3: 3}


def spectrogram(name, start, end, delay):
    audio, _ = sf.read(f"data/audio/{name}", start=start, stop=min(start + SAMPLES - delay, end), dtype="float32")
    audio = np.pad(audio, (delay, SAMPLES - delay - len(audio)))
    spec = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))[:256]
    spec = librosa.amplitude_to_db(spec, ref=np.max, top_db=80)
    return torch.nn.functional.adaptive_avg_pool2d(torch.tensor(spec)[None], (128, 256))[0].numpy().astype(np.float16)


if __name__ == "__main__":
    annotations = pd.read_csv("data/Annotations.csv")
    info = pd.read_csv("data/FileInfo.csv", usecols=range(6))
    data = annotations[annotations.Context.isin(CONTEXTS)].merge(info, on="FileID")
    data["label"] = data.Context.map(CONTEXTS)
    data["day"] = data["Recording time"].str[:10]
    data["hour"] = data["Recording time"].str[11:13].astype(int)
    days = sorted(data.day.unique())
    cut = int(0.9 * len(days))
    data["fold"] = data.day.map({day: i * 5 // cut if i < cut else -1 for i, day in enumerate(days)})
    extra = (data.label == 4).sum() - (data.label == 2).sum()
    data = data.drop(data[data.label == 4].sample(extra, random_state=0).index).reset_index(drop=True)
    data["delay"] = 0
    copies = [data[(data.label == label) & (data.fold >= 0)].assign(delay=25000 * i) for label, n in COPIES.items() for i in range(1, n + 1)]
    data = pd.concat([data] + copies, ignore_index=True)

    with ProcessPoolExecutor() as pool:
        x = list(pool.map(spectrogram, data["File name"], data["Start sample"], data["End sample"], data["delay"], chunksize=64))
    np.save("data/x.npy", np.stack(x))
    data[["label", "fold", "hour", "delay"]].to_csv("data/y.csv", index=False)
    print(pd.crosstab(data.label, data.fold))
