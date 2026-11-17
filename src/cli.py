"""명령줄 실행: 음성 파일 하나 또는 폴더 전체 → ARKit 블렌드쉐입 키프레임.

예)
  python -m src.cli samples/audio/tts/ko_01.wav
  python -m src.cli samples/audio/tts --model small --out outputs
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from .align import WhisperAligner
from .audio import frame_rms, load_wav, to_wav
from .export import write_csv, write_json
from .viseme import (apply_rms, build_keys, build_phones, keys_as_dicts, keys_to_frames, merged_groups,
                     phones_as_dicts)

ROOT = Path(__file__).resolve().parent.parent
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac"}


def load_configs() -> tuple[dict, dict]:
    cfg = {lang: json.loads((ROOT / "config" / f"viseme_{lang}.json").read_text(encoding="utf-8")) for lang in ("ko", "en")}
    pipe = json.loads((ROOT / "config" / "pipeline.json").read_text(encoding="utf-8"))
    return cfg, pipe


def load_script() -> dict[str, dict]:
    p = ROOT / "samples" / "script.csv"
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def process(src: Path, aligner: WhisperAligner, out_root: Path, lang: str | None, use_script: bool,
            preview: bool, cfgs: dict, pipe: dict, script: dict) -> dict:
    sid = src.stem
    out = out_root / sid
    out.mkdir(parents=True, exist_ok=True)
    fps = pipe["fps"]
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    wav = to_wav(src, out / f"{sid}_16k.wav")
    samples = load_wav(wav)
    timing["audio"] = time.perf_counter() - t0

    entry = script.get(sid, {})
    hint = entry.get("text") if use_script else None
    # 언어 우선순위: --lang 옵션 > 대본(script.csv)의 lang > Whisper 자동 감지(ko/en 중에서만)
    lang = lang or entry.get("lang") or None
    t0 = time.perf_counter()
    res = aligner.align(wav, language=lang, text=hint)
    timing["align"] = time.perf_counter() - t0
    if res.language not in ("ko", "en"):
        print(f"  [경고] 감지된 언어 '{res.language}'는 지원하지 않아 영어 규칙으로 처리", file=sys.stderr)
        res.language = "en"

    t0 = time.perf_counter()
    boundary_changes: list[dict] = []
    if pipe.get("boundary", {}).get("enabled"):
        from .boundary import refine_words

        res, boundary_changes = refine_words(res, samples, pipe["boundary"])
    groups = merged_groups(cfgs, res.language)
    phones = build_phones(res, cfgs)
    keys = build_keys(res, phones, pipe)
    n_frames = int(round(res.duration * fps)) + 1
    frames = keys_to_frames(keys, groups, fps, n_frames, pipe.get("lead_frames", 0))
    rms = frame_rms(samples, fps, n_frames)
    frames, scale = apply_rms(frames, rms, pipe["rms"])
    timing["viseme"] = time.perf_counter() - t0

    # 신뢰도: 단어 확률이 임계값보다 낮으면 사람이 검수해야 할 대사로 표시
    thr = pipe.get("review", {}).get("min_word_prob", 0.6)
    probs = [w.probability for w in res.words]
    low = [{"text": w.text, "start": w.start, "probability": w.probability} for w in res.words if w.probability < thr]
    # 음소가 하나도 안 나온 단어(숫자 "3번"의 3, 기호 등)는 입이 움직이지 않으므로 반드시 검수 대상
    voiced = {p.word_index for p in phones}
    silent = [{"text": w.text, "start": w.start} for i, w in enumerate(res.words) if i not in voiced]
    confidence = {
        "min_word_prob": round(min(probs), 3) if probs else 0.0,
        "mean_word_prob": round(sum(probs) / len(probs), 3) if probs else 0.0,
        "threshold": thr,
        "needs_review": bool(low) or bool(silent) or not probs,
        "low_words": low,
        "no_phoneme_words": silent,
    }
    meta = {
        "confidence": confidence,
        "lead_frames": pipe.get("lead_frames", 0),
        "source": src.name, "language": res.language, "fps": fps, "duration": res.duration,
        "aligner": res.aligner, "transcript": res.transcript, "script_hint": bool(hint),
        "language_source": "option/script" if lang else "auto",
        "boundary_refine": bool(pipe.get("boundary", {}).get("enabled")),
        "boundary_changes": boundary_changes,
    }
    words = [w.__dict__ for w in res.words]
    write_json(out / f"{sid}.json", meta, words, keys_as_dicts(keys), frames, phones_as_dicts(phones))
    write_csv(out / f"{sid}.csv", frames, fps)

    if preview:
        from .preview import render_curves, render_video

        t0 = time.perf_counter()
        render_curves(out / f"{sid}_curves.png", frames, rms, res, fps, title=sid)
        render_video(out / f"{sid}_preview.mp4", frames, res, wav, fps, title=sid, keys=keys_as_dicts(keys))
        timing["preview"] = time.perf_counter() - t0

    gen = timing["audio"] + timing["align"] + timing["viseme"]
    flag = f"  ⚠ 검수 필요: {' '.join(w['text'] for w in low) or '단어 없음'}" if confidence["needs_review"] else ""
    print(f"  {sid}: [{res.language}] '{res.transcript}'  단어 {len(res.words)}개, 키 {len(keys)}개, "
          f"{n_frames}프레임 | 생성 {gen:.2f}s (정렬 {timing['align']:.2f}s)"
          + (f" + 미리보기 {timing['preview']:.2f}s" if preview else "") + flag)
    return {"id": sid, "language": res.language, "duration": res.duration, "transcript": res.transcript,
            "words": len(res.words), "min_word_prob": confidence["min_word_prob"],
            "needs_review": confidence["needs_review"], **{f"t_{k}": round(v, 3) for k, v in timing.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="음성 → ARKit 52 블렌드쉐입 립싱크 키프레임")
    ap.add_argument("input", type=Path, help="음성 파일 또는 폴더")
    ap.add_argument("--out", type=Path, default=None,
                    help="출력 폴더. 생략하면 outputs/<입력 폴더 이름> (예: samples/audio/tts → outputs/tts)")
    ap.add_argument("--model", default="medium", help="Whisper 모델 크기 (tiny/base/small/medium/large-v3). 기본 medium: small은 한국어 TTS를 일본어로 오감지함")
    ap.add_argument("--lang", choices=["ko", "en"], default=None, help="생략하면 script.csv의 lang, 없으면 자동 감지")
    ap.add_argument("--use-script", action="store_true", help="samples/script.csv의 대본을 인식 힌트로 사용")
    ap.add_argument("--no-preview", action="store_true", help="MP4/그래프 생성 생략")
    ap.add_argument("--refine-boundaries", action=argparse.BooleanOptionalAction, default=None,
                    help="음량으로 단어 앞뒤 무음을 잘라 낸다. 기본은 config/pipeline.json의 boundary.enabled(켜짐). "
                         "보정 전 결과가 필요하면 --no-refine-boundaries")
    args = ap.parse_args(argv)

    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in AUDIO_EXT) if args.input.is_dir() else [args.input]
    if not files:
        print("처리할 음성 파일이 없습니다.", file=sys.stderr)
        return 1

    if args.out is None:
        tag = args.input.name if args.input.is_dir() else args.input.parent.name
        args.out = ROOT / "outputs" / tag
    cfgs, pipe = load_configs()
    script = load_script()
    t0 = time.perf_counter()
    aligner = WhisperAligner(args.model)
    print(f"모델 로드: {aligner.name} ({time.perf_counter() - t0:.1f}s), 파일 {len(files)}개")

    if args.refine_boundaries is not None:
        pipe.setdefault("boundary", {})["enabled"] = args.refine_boundaries
    rows = [process(f, aligner, args.out, args.lang, args.use_script, not args.no_preview, cfgs, pipe, script) for f in files]
    if rows:
        summary = args.out / "summary.csv"
        with summary.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"요약: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
