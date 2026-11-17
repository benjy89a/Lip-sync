"""기준선: 음량(RMS)만으로 턱 움직이기 (AI 없음).

많은 게임이 실제로 쓰는 가장 단순한 자동 립싱크. 소리가 크면 입을 벌리고 작으면 닫는다.
Whisper 파이프라인과 같은 형식으로 출력해 `src.evaluate`, `src.compare`로 그대로 비교한다.

- jawOpen = RMS / (상위 95% 값) × max_open, 3프레임 이동평균
- 모음 시작 추정: 평활 RMS가 골짜기(극소)에서 다시 올라가는 지점 (음절 핵 검출 방식)
- 입술 모양(pucker, funnel, press …)은 만들 수 없다 → ㅁ/ㅂ/ㅍ 닫힘 표현 불가

사용:
  python -m src.baseline samples/audio/tts            # → outputs/tts_rms/
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from .align import AlignResult
from .audio import SAMPLE_RATE, frame_rms, load_wav, to_wav
from .cli import AUDIO_EXT, ROOT, load_configs
from .export import write_csv, write_json
from .viseme import ARKIT_52, CH_INDEX


def rms_onsets(rms: np.ndarray, fps: int, min_rise: float = 0.15, min_level: float = 0.2) -> list[float]:
    """정규화 RMS에서 '골짜기 → 상승' 지점을 모음 시작으로 본다."""
    r = np.convolve(rms, np.ones(3) / 3, mode="same")
    out = []
    i, n = 1, len(r)
    while i < n - 1:
        if r[i] <= r[i - 1] and r[i] <= r[i + 1]:          # 극소
            j = i
            while j < n - 1 and r[j + 1] >= r[j]:            # 다음 극대까지 상승
                j += 1
            if r[j] - r[i] >= min_rise and r[j] >= min_level:
                out.append(i / fps)
            i = max(j, i + 1)
        else:
            i += 1
    return out


def process(src: Path, out_root: Path, fps: int, max_open: float) -> dict:
    sid = src.stem
    out = out_root / sid
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    wav = to_wav(src, out / f"{sid}_16k.wav")
    samples = load_wav(wav)
    t_audio = time.perf_counter() - t0

    t0 = time.perf_counter()
    duration = len(samples) / SAMPLE_RATE
    n_frames = int(round(duration * fps)) + 1
    rms = frame_rms(samples, fps, n_frames)
    active = rms[rms > rms.max() * 0.05] if rms.max() > 0 else rms
    norm = np.clip(rms / max(np.percentile(active, 95), 1e-6), 0, 1)
    jaw = np.convolve(norm, np.ones(3) / 3, mode="same") * max_open
    frames = np.zeros((n_frames, 52), dtype=np.float32)
    frames[:, CH_INDEX["jawOpen"]] = np.clip(jaw, 0, 1)
    onsets = rms_onsets(norm, fps)
    phones = [{"start": round(t, 4), "end": round(t, 4), "symbol": "", "kind": "vowel", "group": None,
               "word": "", "word_index": k} for k, t in enumerate(onsets)]
    t_viseme = time.perf_counter() - t0

    meta = {"source": src.name, "language": "", "fps": fps, "duration": round(duration, 3),
            "aligner": "baseline:rms", "transcript": "", "script_hint": False}
    write_json(out / f"{sid}.json", meta, [], [], frames, phones)
    write_csv(out / f"{sid}.csv", frames, fps)

    from .preview import render_curves, render_video

    res = AlignResult(language="-", duration=duration, words=[], transcript="(음량 기준선)", aligner="baseline:rms")
    render_curves(out / f"{sid}_curves.png", frames, rms, res, fps, title=f"{sid} (음량 기준선)")
    render_video(out / f"{sid}_preview.mp4", frames, res, wav, fps, title=f"{sid} (RMS baseline)")
    print(f"  {sid}: 모음 시작 추정 {len(onsets)}개, {n_frames}프레임 | 생성 {t_audio + t_viseme:.2f}s")
    return {"id": sid, "language": "", "duration": round(duration, 3), "transcript": "", "words": 0,
            "t_audio": round(t_audio, 3), "t_align": 0.0, "t_viseme": round(t_viseme, 3)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="기준선: 음량만으로 jawOpen 생성")
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="기본 outputs/<입력 폴더 이름>_rms")
    ap.add_argument("--max-open", type=float, default=0.7, help="가장 큰 소리일 때의 jawOpen (비짐 A와 같은 값)")
    args = ap.parse_args(argv)
    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in AUDIO_EXT) if args.input.is_dir() else [args.input]
    tag = args.input.name if args.input.is_dir() else args.input.parent.name
    out = args.out or ROOT / "outputs" / f"{tag}_rms"
    _, pipe = load_configs()
    rows = [process(f, out, pipe["fps"], args.max_open) for f in files]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
