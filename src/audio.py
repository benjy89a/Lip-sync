"""오디오 입출력과 RMS 계산."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def to_wav(src: Path, dst: Path) -> Path:
    """ffmpeg로 아무 형식(wav, m4a, mp3 ...)을 16kHz 모노 16bit wav로 변환한다."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", str(dst)],
        check=True,
    )
    return dst


def load_wav(path: Path) -> np.ndarray:
    """16kHz 모노 wav를 float32 [-1, 1] 배열로 읽는다."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: 샘플레이트 {sr}Hz (to_wav로 먼저 변환해야 함)")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


def frame_rms(samples: np.ndarray, fps: int, n_frames: int) -> np.ndarray:
    """각 프레임 시각을 중심으로 앞뒤 1프레임씩(총 2프레임 길이) 창의 RMS를 구한다."""
    hop = SAMPLE_RATE / fps
    half = int(hop)  # 창 길이 = 2 * hop (앞뒤 반 프레임씩 겹치게)
    out = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        c = int(round(i * hop))
        seg = samples[max(0, c - half): c + half]
        if seg.size:
            out[i] = float(np.sqrt(np.mean(seg ** 2)))
    return out
