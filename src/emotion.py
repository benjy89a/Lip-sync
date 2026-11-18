"""감정 레이어: 사람이 주는 감정 가이드 → 얼굴 전체 ARKit 커브.

가이드 형식: "화남:0.6", "anger:0.6", "슬픔:0.5+화남:0.3" (강도 생략 시 1.0)

1. 가이드의 감정 프리셋(FACS 기본 감정 → ARKit 값, config/emotion.json)을 강도만큼 섞는다.
2. 말이 시작될 때 서서히 올라가고, 끝나면 rest_level까지 내려가는 강도 곡선을 곱한다.
3. 입 레이어(발음)와 합친다.
   - 눈·눈썹·볼·코: 감정 값을 그대로 쓴다 (발음 레이어에는 값이 없다).
   - 입 주변: 발음이 우선. 감정 값을 emotion_weight만큼 줄여서 더한다.
   - 입술이 닫힌 프레임(ㅁ ㅂ ㅍ)에서는 감정이 입을 벌리거나 옆으로 당기지 못한다.
   - 입술을 오므리는 발음 중에는 감정의 웃음·당김을 그만큼 줄인다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .align import AlignResult
from .viseme import ARKIT_52, CH_INDEX, expand_channels

ROOT = Path(__file__).resolve().parent.parent
UPPER_PREFIX = ("brow", "eye", "cheek", "nose")


def load_emotion_config() -> dict:
    return json.loads((ROOT / "config" / "emotion.json").read_text(encoding="utf-8"))


def parse_guide(guide: str | None, cfg: dict) -> list[tuple[str, float]]:
    """'슬픔:0.5+화남:0.3' → [('sadness', 0.5), ('anger', 0.3)]"""
    if not guide or not guide.strip():
        return []
    out = []
    for part in re.split(r"[+,]", guide):
        part = part.strip()
        if not part:
            continue
        name, _, amt = part.partition(":")
        name = name.strip()
        key = cfg["aliases"].get(name, name.lower())
        if key not in cfg["presets"]:
            raise ValueError(f"알 수 없는 감정 '{name}'. 사용 가능: {', '.join(cfg['presets'])} 또는 {', '.join(cfg['aliases'])}")
        out.append((key, float(amt) if amt.strip() else 1.0))
    return out


def guide_label(parts: list[tuple[str, float]], cfg: dict) -> str:
    if not parts:
        return "중립"
    return " + ".join(f"{cfg['presets'][k]['label']} {v:.1f}" for k, v in parts)


def emotion_vector(parts: list[tuple[str, float]], cfg: dict) -> np.ndarray:
    """감정 프리셋들을 강도만큼 섞은 52채널 값 (말하는 중 배율 적용 전)."""
    vec = np.zeros(52, dtype=np.float32)
    for key, amt in parts:
        for ch, v in expand_channels(cfg["presets"][key]["values"]).items():
            vec[CH_INDEX[ch]] += v * amt
    return np.clip(vec, 0, 1)


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def speaking(result: AlignResult, fps: int, n_frames: int, edge_sec: float = 0.15) -> np.ndarray:
    """말하는 구간이면 1, 아니면 0 (가장자리는 부드럽게)."""
    t = np.arange(n_frames) / fps
    if not result.words:
        return np.zeros(n_frames, dtype=np.float32)
    s, e = result.words[0].start, result.words[-1].end
    return (_smooth((t - (s - edge_sec)) / edge_sec) * (1 - _smooth((t - e) / edge_sec))).astype(np.float32)


def envelope(result: AlignResult, fps: int, n_frames: int, env: dict) -> np.ndarray:
    """프레임별 감정 강도 (0~1). 처음부터 rest_level, 말하는 동안 1.0, 끝나면 다시 rest_level."""
    t = np.arange(n_frames) / fps
    rest = env["rest_level"]
    if not result.words:
        return np.full(n_frames, rest, dtype=np.float32)
    s = max(0.0, result.words[0].start - env["pre_roll_sec"])
    e = result.words[-1].end
    up = _smooth((t - s) / env["attack_sec"])
    down = _smooth((t - e) / env["release_sec"])
    return (rest + (1 - rest) * up * (1 - down)).astype(np.float32)


def emotion_frames(parts: list[tuple[str, float]], result: AlignResult, fps: int, n_frames: int, cfg: dict) -> np.ndarray:
    vec = emotion_vector(parts, cfg) * cfg["speech_scale"]
    return envelope(result, fps, n_frames, cfg["envelope"])[:, None] * vec[None, :]


def merge(mouth: np.ndarray, emo: np.ndarray, cfg: dict, talk: np.ndarray | None = None) -> np.ndarray:
    """입 레이어(발음)와 감정 레이어를 합친다. talk: 말하는 구간(0~1). 말하는 동안만 입 주변 감정을 줄인다."""
    mb = cfg["mouth_blend"]
    out = mouth.copy()
    upper = np.array([n.startswith(UPPER_PREFIX) for n in ARKIT_52])
    out[:, upper] = np.maximum(out[:, upper], emo[:, upper])

    e = emo.copy()
    if talk is None:
        talk = np.ones(len(mouth), dtype=np.float32)
    weight = 1 - (1 - mb["emotion_weight"]) * talk
    e[:, ~upper] *= weight[:, None]
    # 입술을 오므리는 발음 중에는 감정의 웃음·당김을 줄인다
    for base, against in mb["antagonists"].items():
        if base.startswith("_"):
            continue
        a = np.zeros(len(mouth), dtype=np.float32)
        for name in against:
            for ch in expand_channels({name: 1}).keys():
                a = np.maximum(a, mouth[:, CH_INDEX[ch]])
        for ch in expand_channels({base: 1}).keys():
            e[:, CH_INDEX[ch]] *= np.clip(1 - a, 0, 1)
    # 입술이 닫힌 프레임에서는 감정이 입을 벌리거나 당기지 못한다
    cl = mb["closure"]
    press = (mouth[:, CH_INDEX["mouthPressLeft"]] + mouth[:, CH_INDEX["mouthPressRight"]]) / 2
    closed = (mouth[:, CH_INDEX["jawOpen"]] <= cl["jaw_max"]) & (press >= cl["press_min"])
    for ch in cl["suppress"]:
        e[closed, CH_INDEX[ch]] = 0
    out[:, ~upper] += e[:, ~upper]
    return np.clip(out, 0, 1)
