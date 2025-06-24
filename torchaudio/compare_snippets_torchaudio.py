#!/usr/bin/env python
"""
compare_snippets_torchaudio.py
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
• Python ≥ 3.13  
• Torchaudio ≥ 2.7 (dispatcher mode)  
• Requires a dispatcher backend: `pip install soundfile` or `brew install ffmpeg`
"""

from __future__ import annotations
import os, json, uuid
from pathlib import Path
from collections import defaultdict

import torch, torch.nn.functional as F
import torchaudio, torchaudio.transforms as T
from pydub import AudioSegment
#from torchaudio import pipelines as TAP
# --------------------------------------------------------------------- #
# 0.  Choose & validate the torchaudio I/O backend                      #
# --------------------------------------------------------------------- #
BACKEND = "soundfile"        # or "ffmpeg"
if BACKEND not in torchaudio.list_audio_backends():
    raise RuntimeError(
        f"{BACKEND!r} backend not available – run "
        "`pip install soundfile` or `brew install ffmpeg` in the venv."
    )

# --------------------------------------------------------------------- #
# 1.  Load Wav2Vec2                                                     #
# --------------------------------------------------------------------- #
print("Loading Wav2Vec2_BASE …")
bundle = torchaudio.pipelines.WAV2VEC2_BASE
#bundle = TAP.MERT_V1_BASE

model  = bundle.get_model().eval()
print("Model ready.\n")

# --------------------------------------------------------------------- #
# 2.  Helpers                                                           #
# --------------------------------------------------------------------- #
def parse_ts(ts: str) -> float:
    m, s = map(int, ts.strip().split(":"))
    return m * 60 + s

def carve_snippet(mp3: str, rng: str) -> str:
    start, end = map(str.strip, rng.split("-"))
    audio = AudioSegment.from_mp3(mp3)
    segment = audio[int(parse_ts(start)*1000): int(parse_ts(end)*1000)]
    out = f"temp_{uuid.uuid4()}.wav"
    segment.export(out, format="wav")
    return out

def embed(mp3: str, rng: str) -> torch.Tensor | None:
    tmp = None
    try:
        tmp, (wav, sr) = carve_snippet(mp3, rng), (None, None)
        wav, sr = torchaudio.load(tmp, backend=BACKEND)
        if sr != bundle.sample_rate:
            wav = T.Resample(sr, bundle.sample_rate)(wav)
        feats, _ = model.extract_features(wav)
        if isinstance(feats, list):
            feats = feats[-1]                # deepest transformer layer
        return feats.mean(dim=1).squeeze()   # (768,)
    except Exception as e:
        print(f"[embed] {Path(mp3).name}: {e}")
        return None
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

def cosine(v1: torch.Tensor | None, v2: torch.Tensor | None) -> float:
    if v1 is None or v2 is None: return 0.0
    v1, v2 = v1.flatten(), v2.flatten()
    return float(torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-12))

# --------------------------------------------------------------------- #
# 3.  Load annotations and build song-to-mate map                       #
# --------------------------------------------------------------------- #
ann = json.load(open("data/annotations_armon.txt"))
mate: dict[str, str] = {}
for p in ann:
    mate[p["song1"]] = p["song2"]
    mate[p["song2"]] = p["song1"]

# unique songs with their own snippet ranges
songs = {p["song1"]: p["snippet1"] for p in ann} | {p["song2"]: p["snippet2"] for p in ann}
print(f"Found {len(songs)} unique tracks.\nEmbedding …")

# --------------------------------------------------------------------- #
# 4.  Compute embeddings once                                           #
# --------------------------------------------------------------------- #
emb: dict[str, torch.Tensor] = {s: embed(s, rng) for s, rng in songs.items()}

# --------------------------------------------------------------------- #
# 5.  All-vs-all similarity matrix                                      #
# --------------------------------------------------------------------- #
sim: dict[str, dict[str, float]] = defaultdict(dict)
track_list = list(emb.keys())
for i, s1 in enumerate(track_list):
    for s2 in track_list[i+1:]:
        score = cosine(emb[s1], emb[s2])
        sim[s1][s2] = sim[s2][s1] = score

# --------------------------------------------------------------------- #
# 6.  For each song → top-4 matches & success flag                      #
# --------------------------------------------------------------------- #
hits = 0
for s in track_list:
    top4 = sorted(sim[s].items(), key=lambda kv: kv[1], reverse=True)[:4]
    top4_names = [Path(t).name for t, _ in top4]
    success = mate[s] in (t for t, _ in top4)
    hits += success
    print(f"{Path(s).name:<35}  →  {', '.join(top4_names)}   "
          f"[{'✔' if success else '✘'}]")

# --------------------------------------------------------------------- #
# 7.  Overall accuracy                                                  #
# --------------------------------------------------------------------- #
print("\n" + "-"*60)
total = len(track_list)
print(f"Top-4 retrieval success: {hits}/{total}  "
      f"({hits/total:.1%})")
