"""음소 → 비짐 키 → ARKit 52 프레임 커브.

1. 단어 구간을 음소 가중치(모음이 큼)대로 나누고, 각 음소 구간 중앙에 비짐 키를 둔다.
2. 그룹이 없는 자음(ㄱ ㄴ ㄷ ...)은 키를 만들지 않아 앞뒤 모음 값을 보간으로 이어받는다.
3. ㅁ/ㅂ/ㅍ(MBP)는 닫힘이 보이도록 키 중앙 앞뒤로 같은 값을 유지한다.
4. 단어 사이 공백이 길면 REST(입 다묾) 키를 넣는다.
5. 키 사이는 smoothstep으로 보간한다. 전체 스무딩 필터는 MBP 닫힘을 뭉개므로 쓰지 않는다.
6. 오디오 RMS로 jawOpen 등 일부 채널의 세기를 조절한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .align import AlignResult
from .phonemes import word_units

ARKIT_52 = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft",
    "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight",
    "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]
CH_INDEX = {n: i for i, n in enumerate(ARKIT_52)}
assert len(ARKIT_52) == 52


def expand_channels(values: dict[str, float]) -> dict[str, float]:
    """'mouthSmile' 처럼 Left/Right 쌍의 기본 이름은 양쪽 채널에 같은 값으로 펼친다."""
    out: dict[str, float] = {}
    for name, v in values.items():
        if name in CH_INDEX:
            out[name] = v
        elif name + "Left" in CH_INDEX and name + "Right" in CH_INDEX:
            out[name + "Left"] = v
            out[name + "Right"] = v
        else:
            raise KeyError(f"ARKit 52에 없는 채널: {name}")
    return out


def group_vector(group: str, groups: dict[str, dict]) -> np.ndarray:
    vec = np.zeros(52, dtype=np.float32)
    for name, v in expand_channels(groups[group]).items():
        vec[CH_INDEX[name]] = v
    return vec


@dataclass
class Key:
    time: float
    group: str
    symbol: str
    word: str


@dataclass
class Phone:
    """단어 구간 안에서 음소 하나가 차지하는 시간. 평가(모음 시작 오차, 입 닫힘 구간)에 쓴다."""
    start: float
    end: float
    symbol: str
    kind: str
    group: str | None
    word: str
    word_index: int


def build_phones(result: AlignResult, cfgs: dict[str, dict]) -> list[Phone]:
    """단어 구간을 음소 가중치(모음이 큼)대로 나눈다."""
    phones: list[Phone] = []
    for wi, w in enumerate(result.words):
        units = word_units(w.text, result.language, cfgs)
        total = sum(u.weight for u in units)
        if total <= 0 or w.end <= w.start:
            continue
        t = w.start
        span = w.end - w.start
        for u in units:
            d = span * u.weight / total
            phones.append(Phone(t, t + d, u.symbol, u.kind, u.group, w.text, wi))
            t += d
    return phones


def build_keys(result: AlignResult, phones: list[Phone], pipe: dict) -> list[Key]:
    keys: list[Key] = []
    gap = pipe["silence_gap_sec"]
    pad = pipe["rest_pad_sec"]
    hold = pipe["mbp_hold_sec"]

    keys.append(Key(0.0, "REST", "", ""))
    prev_end = 0.0
    for w in result.words:
        # 앞 단어와 공백이 길면 말이 끝난 직후 입을 다물고, 다음 단어 직전까지 다문 상태를 유지한다.
        # (첫 단어 앞은 0초의 REST 키가 이미 있으므로 "직전" 키만 넣는다)
        if w.start - prev_end > gap:
            if prev_end > 0:
                keys.append(Key(prev_end + pad, "REST", "", ""))
            keys.append(Key(max(w.start - pad, prev_end + pad if prev_end > 0 else 0.0), "REST", "", ""))
        prev_end = max(prev_end, w.end)
    for p in phones:
        if p.group is None:
            continue
        c = (p.start + p.end) / 2
        if p.group == "MBP":
            h = min(hold, (p.end - p.start) / 3)
            keys.append(Key(c - h, p.group, p.symbol, p.word))
            keys.append(Key(c + h, p.group, p.symbol, p.word))
        else:
            keys.append(Key(c, p.group, p.symbol, p.word))
    keys.append(Key(prev_end + pad, "REST", "", ""))
    keys.append(Key(max(result.duration, prev_end + pad), "REST", "", ""))
    keys.sort(key=lambda k: k.time)
    return keys


def phones_as_dicts(phones: list[Phone]) -> list[dict]:
    return [{**asdict(p), "start": round(p.start, 4), "end": round(p.end, 4)} for p in phones]


def _smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3 - 2 * x)


def keys_to_frames(keys: list[Key], groups: dict[str, dict], fps: int, n_frames: int, lead_frames: int = 0) -> np.ndarray:
    times = np.array([k.time for k in keys]) - lead_frames / fps
    vecs = np.stack([group_vector(k.group, groups) for k in keys])
    ft = np.arange(n_frames) / fps
    idx = np.clip(np.searchsorted(times, ft, side="right") - 1, 0, len(keys) - 1)
    nxt = np.clip(idx + 1, 0, len(keys) - 1)
    span = times[nxt] - times[idx]
    x = np.where(span > 1e-6, (ft - times[idx]) / np.where(span > 1e-6, span, 1), 0.0)
    x = _smoothstep(np.clip(x, 0, 1))[:, None]
    frames = vecs[idx] * (1 - x) + vecs[nxt] * x
    return frames.astype(np.float32)


def apply_rms(frames: np.ndarray, rms: np.ndarray, rms_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """음량에 따라 입 벌림 채널을 키우거나 줄인다. 반환: (조정된 프레임, 프레임별 배율)."""
    active = rms[rms > rms.max() * 0.05] if rms.max() > 0 else rms
    ref = np.percentile(active, rms_cfg["percentile"]) if active.size else 1.0
    norm = np.clip(rms / max(ref, 1e-6), 0, None)
    floor = rms_cfg["floor"]
    scale = np.clip(floor + (1 - floor) * norm, 0, rms_cfg["max_scale"])
    out = frames.copy()
    for name in rms_cfg["channels"]:
        out[:, CH_INDEX[name]] *= scale
    return np.clip(out, 0, 1), scale


def merged_groups(cfgs: dict[str, dict], lang: str) -> dict[str, dict]:
    """주 언어 그룹을 우선하고, 다른 언어에만 있는 그룹(FV, TH ...)을 합친다."""
    other = "en" if lang == "ko" else "ko"
    return {**cfgs[other]["groups"], **cfgs[lang]["groups"]}


def keys_as_dicts(keys: list[Key]) -> list[dict]:
    return [{**asdict(k), "time": round(k.time, 4)} for k in keys]
