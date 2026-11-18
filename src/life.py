"""생기 레이어: 눈 깜빡임과 시선 움직임.

음성에는 눈에 대한 정보가 거의 없으므로, 사람의 일반적인 패턴을 규칙으로 만든다.
- 깜빡임: 평균 간격(초)을 중심으로 흔들리는 간격. 말을 시작하기 직전에 한 번 깜빡인다.
- 시선: 짧은 간격으로 작은 범위 안에서 빠르게 옮겨 가고(도약 안구 운동), 그 사이에는 머문다.
같은 파일이면 항상 같은 결과가 나오도록 파일 이름에서 만든 고정 난수를 쓴다 (재현성).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .align import AlignResult
from .viseme import CH_INDEX

ROOT = Path(__file__).resolve().parent.parent


def load_life_config() -> dict:
    return json.loads((ROOT / "config" / "life.json").read_text(encoding="utf-8"))


def _rng(seed_text: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16))


def _blink_shape(t: np.ndarray, t0: float, b: dict) -> np.ndarray:
    """t0에 시작하는 깜빡임 하나 (0~1). 빨리 감고, 잠깐 머물고, 천천히 뜬다."""
    c, h, o = b["close_sec"], b["hold_sec"], b["open_sec"]
    x = t - t0
    out = np.zeros_like(t)
    m1 = (x >= 0) & (x < c)
    out[m1] = np.sin(np.pi / 2 * x[m1] / c)
    m2 = (x >= c) & (x < c + h)
    out[m2] = 1
    m3 = (x >= c + h) & (x < c + h + o)
    out[m3] = np.cos(np.pi / 2 * (x[m3] - c - h) / o)
    return out


def life_frames(result: AlignResult, fps: int, n_frames: int, cfg: dict, seed_text: str) -> np.ndarray:
    rng = _rng(seed_text)
    t = np.arange(n_frames) / fps
    dur = n_frames / fps
    out = np.zeros((n_frames, 52), dtype=np.float32)

    # 깜빡임
    b = cfg["blink"]
    times = []
    if result.words:
        times.append(max(0.0, result.words[0].start - b["before_speech_sec"]))
    x = rng.uniform(0.3, b["interval_mean_sec"])
    while x < dur:
        if all(abs(x - y) > b["min_gap_sec"] for y in times):
            times.append(x)
        x += max(b["min_gap_sec"], rng.normal(b["interval_mean_sec"], b["interval_sd_sec"]))
    blink = np.zeros(n_frames, dtype=np.float32)
    for t0 in sorted(times):
        blink = np.maximum(blink, _blink_shape(t, t0, b))
    out[:, CH_INDEX["eyeBlinkLeft"]] = blink
    out[:, CH_INDEX["eyeBlinkRight"]] = blink

    # 시선: 목표점(가로 gx, 세로 gy, -1~1)으로 빠르게 옮겨 가고 머문다
    g = cfg["gaze"]
    gx = np.zeros(n_frames, dtype=np.float32)
    gy = np.zeros(n_frames, dtype=np.float32)
    cur = np.array([0.0, 0.0])
    x = 0.0
    while x < dur:
        nxt = np.array([rng.uniform(-g["range_x"], g["range_x"]), rng.uniform(-g["range_y"], g["range_y"])])
        a, e = int(x * fps), int(min(dur, x + g["saccade_sec"]) * fps)
        hold_end = int(min(dur, x + rng.uniform(g["hold_min_sec"], g["hold_max_sec"])) * fps)
        for i in range(a, min(n_frames, hold_end + 1)):
            k = 1.0 if i >= e else (i - a) / max(1, e - a)
            gx[i], gy[i] = cur + (nxt - cur) * k
        cur = nxt
        x = hold_end / fps
    # 가로: +는 캐릭터의 왼쪽을 봄 → 왼눈은 바깥(Out), 오른눈은 안쪽(In)
    left_side, right_side = np.clip(gx, 0, None), np.clip(-gx, 0, None)
    out[:, CH_INDEX["eyeLookOutLeft"]] = left_side
    out[:, CH_INDEX["eyeLookInRight"]] = left_side
    out[:, CH_INDEX["eyeLookInLeft"]] = right_side
    out[:, CH_INDEX["eyeLookOutRight"]] = right_side
    up, down = np.clip(gy, 0, None), np.clip(-gy, 0, None)
    for side in ("Left", "Right"):
        out[:, CH_INDEX[f"eyeLookUp{side}"]] = up
        out[:, CH_INDEX[f"eyeLookDown{side}"]] = down
    return out
