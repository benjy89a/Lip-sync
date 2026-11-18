"""JSON / CSV 출력. 재현성 확인을 위해 생성 시각 같은 실행마다 바뀌는 값은 넣지 않는다."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .viseme import ARKIT_52


def _curves(frames: np.ndarray, nonzero_only: bool = False) -> dict:
    return {name: [round(float(v), 4) for v in frames[:, i]] for i, name in enumerate(ARKIT_52)
            if not nonzero_only or frames[:, i].max() > 0}


def write_json(path: Path, meta: dict, words: list[dict], keys: list[dict], frames: np.ndarray,
               phones: list[dict] | None = None, layers: dict[str, np.ndarray] | None = None) -> None:
    data = {
        "meta": {**meta, "channels": ARKIT_52, "frame_count": int(frames.shape[0])},
        "words": words,
        "phones": phones or [],  # 음소별 시간 구간 (평가용)
        "keys": keys,
        # 채널별 커브: DCC 툴에서 채널 단위로 키를 넣기 쉽도록 채널 → 프레임 값 배열 (모든 레이어를 합친 최종값)
        "curves": _curves(frames),
    }
    if layers:
        # 레이어별 값 (합치기 전). 애니메이터가 "감정만 약하게" 같은 수정을 할 수 있도록 따로 남긴다. 값이 있는 채널만.
        data["layers"] = {name: _curves(arr, nonzero_only=True) for name, arr in layers.items()}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def write_csv(path: Path, frames: np.ndarray, fps: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time", *ARKIT_52])
        for i, row in enumerate(frames):
            w.writerow([i, f"{i / fps:.4f}", *(f"{v:.4f}" for v in row)])
