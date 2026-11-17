"""음량으로 단어 경계 보정 (추가 시도).

Whisper의 단어 시각은 빈틈 없이 이어진다. 그래서
  - 쉼이 앞 단어에 흡수되어 음소가 늘어지고 (예: 2음절 "먹자"가 0.74초),
  - 첫 단어는 소리가 나기 전부터 시작한다고 잡힌다 (예: 0.00초 시작, 실제 소리는 0.13초부터).
10ms 단위 음량으로 "소리가 나는 구간"을 찾아, 단어 앞뒤에 붙은 무음만 잘라 낸다.

건드리지 않는 것:
  - 단어 중간의 무음: ㅂ ㄱ ㄷ 같은 파열음은 소리가 나기 전에 조음 기관이 닫힌 무음 구간이 있다.
  - 단어 앞은 pre_roll_sec만큼 여유를 남긴다 (같은 이유: 첫 자음의 닫힘 구간).
  - Whisper가 단어를 실제보다 늦게 시작한 경우(앞으로 늘리기)는 다루지 않는다.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .align import AlignResult
from .audio import SAMPLE_RATE

HOP_SEC = 0.01


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True가 이어지는 구간들의 (시작, 끝) 인덱스. 끝은 포함하지 않는다."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def activity(samples: np.ndarray, rel_threshold: float, fill_gap_sec: float = 0.08, min_active_sec: float = 0.06) -> np.ndarray:
    """10ms 단위로 소리가 나는지 여부. 기준 = 큰 소리(상위 95%)의 rel_threshold 배.

    fill_gap_sec보다 짧은 끊김은 메우고, 그 뒤에도 min_active_sec보다 짧은 소리는 잡음(입 소리, 클릭)으로 보고 버린다.
    """
    hop = int(SAMPLE_RATE * HOP_SEC)
    n = max(1, len(samples) // hop)
    rms = np.array([np.sqrt(np.mean(samples[max(0, i * hop - hop // 2): i * hop + hop + hop // 2] ** 2) + 1e-12)
                    for i in range(n)])
    loud = rms[rms > rms.max() * 0.05] if rms.max() > 0 else rms
    ref = np.percentile(loud, 95) if loud.size else 1.0
    act = rms > ref * rel_threshold
    # 짧은 끊김은 메운다 (파열음의 닫힘 구간을 무음으로 보지 않기 위해)
    max_gap = int(round(fill_gap_sec / HOP_SEC))
    i = 0
    while i < n:
        if not act[i]:
            j = i
            while j < n and not act[j]:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                act[i:j] = True
            i = j
        else:
            i += 1
    for a, b in _runs(act):
        if (b - a) * HOP_SEC < min_active_sec:
            act[a:b] = False
    return act


def refine_words(result: AlignResult, samples: np.ndarray, cfg: dict) -> tuple[AlignResult, list[dict]]:
    """단어 앞뒤의 무음을 잘라 낸 새 AlignResult와 변경 내역을 돌려준다."""
    act = activity(samples, cfg["rel_threshold"], min_active_sec=cfg.get("min_active_sec", 0.06))
    pre, post, min_change = cfg["pre_roll_sec"], cfg["post_roll_sec"], cfg["min_change_sec"]
    long_gap = int(round(cfg.get("long_gap_sec", 0.2) / HOP_SEC))
    words, changes = [], []
    for w in result.words:
        a, b = int(round(w.start / HOP_SEC)), min(len(act), int(round(w.end / HOP_SEC)))
        seg = act[a:b]
        if b <= a or not seg.any():
            words.append(w)
            continue
        # 단어 안에 긴 무음(long_gap_sec 이상)이 있으면 소리 구간을 덩어리로 나누고, 소리가 가장 긴 덩어리만 이 단어로 본다.
        # (파열음의 닫힘은 이보다 짧다. 긴 무음 뒤의 짧은 소리는 대개 다음 단어의 앞부분이 걸친 것)
        groups: list[list[int]] = []  # [시작, 끝, 소리 길이]
        for ra, rb in _runs(seg):
            if groups and ra - groups[-1][1] < long_gap:
                groups[-1][1] = rb
                groups[-1][2] += rb - ra
            else:
                groups.append([ra, rb, rb - ra])
        best = max(groups, key=lambda g: g[2])
        first, last = a + best[0], a + best[1] - 1
        new_s = max(w.start, first * HOP_SEC - pre)
        new_e = min(w.end, (last + 1) * HOP_SEC + post)
        if new_s - w.start < min_change:
            new_s = w.start
        if w.end - new_e < min_change:
            new_e = w.end
        if new_e - new_s < 0.08 or (new_s == w.start and new_e == w.end):
            words.append(w)
            continue
        words.append(replace(w, start=round(new_s, 3), end=round(new_e, 3)))
        changes.append({"text": w.text, "start": [w.start, round(new_s, 3)], "end": [w.end, round(new_e, 3)]})
    return replace(result, words=words), changes


def silent_open_frames(samples: np.ndarray, jaw: list[float], fps: int, rel_threshold: float, jaw_min: float = 0.1) -> int:
    """무음인데 입이 열려 있는 프레임 수 (라벨 없이 볼 수 있는 지표)."""
    act = activity(samples, rel_threshold)
    count = 0
    for i, v in enumerate(jaw):
        k = int(round(i / fps / HOP_SEC))
        lo, hi = max(0, k - 2), min(len(act), k + 3)  # 프레임 주변 ±20ms가 모두 무음일 때만 센다
        if hi > lo and not act[lo:hi].any() and v > jaw_min:
            count += 1
    return count
