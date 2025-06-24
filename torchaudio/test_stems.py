#!/usr/bin/env python
"""
compare_snippets_stems.py
──────────────────────────────────────────────────────────────────────
• torchaudio ≥ 2.8-dev (gives HDEMUCS_HIGH_MUSDB_PLUS + WavLM_LARGE)
• torch ≥ 2.8-dev
• soundfile or ffmpeg backend installed

Evaluates TOP-4 recall on data/annotations_armon.txt using
Hybrid-Demucs-PLUS  (4 stems)  +  WavLM-Large embeddings.
"""

from __future__ import annotations
import os, json, uuid, warnings
from pathlib import Path
from collections import defaultdict

import torch, torch.nn.functional as F
import torchaudio, torchaudio.transforms as T
from pydub import AudioSegment

# ────────── config ──────────────────────────────────────────────── #
BACKEND       = "soundfile"                 # or "ffmpeg"
STEM_WEIGHTS  = dict(vocals=0.3,
                     bass   =0.1,
                     drums  =0.1,
                     other  =0.4)
ANNOTATIONS   = Path("data/annotations_armon.txt")
# ------------------------------------------------------------------ #

if BACKEND not in torchaudio.list_audio_backends():
    raise RuntimeError("No I/O backend – run `pip install soundfile` or `brew install ffmpeg`")

# ────────── bundles ─────────────────────────────────────────────── #
from torchaudio import pipelines as TAP
DEMUCS_B   = TAP.HDEMUCS_HIGH_MUSDB_PLUS     # 44 100 Hz, 4 stems
WAVLM_B    = TAP.WAVLM_LARGE                # 16 000 Hz, 1024-d

demucs     = DEMUCS_B.get_model().eval()
wavlm      = WAVLM_B.get_model().eval()
STEM_NAMES = list(demucs.sources)           # ['vocals','drums','bass','other']

print("Loaded Hybrid-Demucs-PLUS  +  WavLM-Large\n")

# ────────── helpers ─────────────────────────────────────────────── #
def parse(ts: str) -> float:
    m, s = map(int, ts.strip().split(":")); return m*60 + s

def carve(mp3: str, span: str) -> str:
    a, b = map(str.strip, span.split("-"))
    seg  = AudioSegment.from_mp3(mp3)
    seg  = seg[int(parse(a)*1000): int(parse(b)*1000)]
    fn   = f"tmp_{uuid.uuid4()}.wav"
    seg.export(fn, format="wav"); return fn

@torch.inference_mode()
def separate_stems(wav: torch.Tensor, sr: int) -> dict[str, torch.Tensor]:
    """wav: (B,C,T)  →  dict{name: (T,)}  (mono stems)"""
    if sr != DEMUCS_B.sample_rate:
        wav = T.Resample(sr, DEMUCS_B.sample_rate)(wav)
    out = demucs(wav)                       # (B,S,C,T)
    stems = {}
    for i, name in enumerate(STEM_NAMES):
        stems[name] = out[:, i].mean(1).squeeze()   # (T,)
    return stems

@torch.inference_mode()
def wavlm_vec(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """mono (T,) → 1024-d L2-normalised embedding"""
    if sr != WAVLM_B.sample_rate:
        wav = T.Resample(sr, WAVLM_B.sample_rate)(wav)
    feats, _ = wavlm.extract_features(wav.unsqueeze(0))   # (1,T)
    feats = torch.stack(feats[4:9]).mean(0)               # avg layers 4-8
    vec   = feats.mean(1).squeeze()                       # (1024,)
    return F.normalize(vec, dim=0)

def embed_snippet(mp3: str, span: str) -> torch.Tensor | None:
    tmp = None
    try:
        tmp = carve(mp3, span)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            mix, sr = torchaudio.load(tmp, backend=BACKEND)   # (C,T)
        if mix.dim() == 1:
            mix = mix.unsqueeze(0)        # mono file
        mix = mix.unsqueeze(0)            # add batch → (1,C,T)

        stems = separate_stems(mix, sr)
        parts = [
            STEM_WEIGHTS[n] * wavlm_vec(a, DEMUCS_B.sample_rate)
            for n, a in stems.items()
            if STEM_WEIGHTS.get(n, 0.0) > 0.0
        ]
        if not parts:
            return None
        return F.normalize(torch.stack(parts).sum(0), dim=0)
    except Exception as e:
        print(f"[embed] {Path(mp3).name}: {e}")
        return None
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

def cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    return float(torch.dot(a, b)) if (a is not None and b is not None) else 0.0

# ────────── embed catalogue ─────────────────────────────────────── #
pairs = json.load(ANNOTATIONS.open())
mate  = {p["song1"]: p["song2"] for p in pairs} | {p["song2"]: p["song1"] for p in pairs}
songs = {p["song1"]: p["snippet1"] for p in pairs} | {p["song2"]: p["snippet2"] for p in pairs}

print(f"Embedding {len(songs)} unique tracks …")
emb = {s: embed_snippet(s, span) for s, span in songs.items()}

# ────────── similarity matrix ───────────────────────────────────── #
sim = defaultdict(dict)
keys = list(emb)
for i, s1 in enumerate(keys):
    for s2 in keys[i+1:]:
        sc = cosine(emb[s1], emb[s2])
        sim[s1][s2] = sim[s2][s1] = sc

# ────────── evaluation ──────────────────────────────────────────── #
hits = 0
for s in keys:
    top4 = sorted(sim[s].items(), key=lambda kv: kv[1], reverse=True)[:4]
    ok   = mate[s] in (t for t,_ in top4)
    hits += ok
    print(f"{Path(s).name:<32} → "
          f"{', '.join(Path(t).name for t,_ in top4)}   "
          f"[{'✔' if ok else '✘'}]")

print("-"*60)
print(f"Top-4 recall (Demucs + WavLM): {hits}/{len(keys)} = {hits/len(keys):.1%}")
