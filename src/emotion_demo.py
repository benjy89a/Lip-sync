"""감정 비교 시연: 같은 음성을 감정 프리셋별로 만들어 한 화면(4×2)에 모은다.

사용:
  python -m src.emotion_demo samples/audio/ko_01.wav
출력: outputs/emotion_demo/<id>/<감정>/<id>.json …, outputs/emotion_demo/<id>_emotions.mp4, <id>_emotions.png
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .cli import ROOT, main as cli_main
from .emotion import load_emotion_config
from .preview import FONT_PATHS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="같은 음성을 감정별로 만들어 비교 영상 생성")
    ap.add_argument("input", type=Path, help="음성 파일 하나")
    ap.add_argument("--amount", type=float, default=0.8, help="감정 강도 (0~1)")
    ap.add_argument("--still-times", type=float, nargs="+", default=[4.6, 6.2],
                    help="정지 화면을 뽑을 시각(초). 기본: 입이 열린 순간, 말이 끝난 뒤")
    args = ap.parse_args(argv)

    cfg = load_emotion_config()
    sid = args.input.stem
    root = ROOT / "outputs" / "emotion_demo" / sid
    names = list(cfg["presets"])  # neutral + 기본 7감정 = 8
    for name in names:
        guide = "" if name == "neutral" else f"{name}:{args.amount}"
        print(f"[{cfg['presets'][name]['label']}]")
        cli_main([str(args.input), "--out", str(root / name), "--emotion", guide, "--life"])

    font = next((p for p in FONT_PATHS if Path(p).exists()), None)
    inputs, filters = [], []
    for i, name in enumerate(names):
        inputs += ["-i", str(root / name / sid / f"{sid}_preview.mp4")]
        label = cfg["presets"][name]["label"] + ("" if name == "neutral" else f" {args.amount}")
        draw = (f",drawtext=fontfile={font}:text='{label}':x=12:y=h-46:fontsize=30:fontcolor=white:"
                f"box=1:boxcolor=black@0.6:boxborderw=8") if font else ""
        # 왼쪽 얼굴 부분만 잘라서 모은다 (오른쪽 막대 패널 제외)
        filters.append(f"[{i}:v]crop=500:460:0:0,scale=400:368{draw}[v{i}]")
    layout = "|".join(f"{(i % 4) * 400}_{(i // 4) * 368}" for i in range(len(names)))
    fc = ";".join(filters) + ";" + "".join(f"[v{i}]" for i in range(len(names))) + f"xstack=inputs={len(names)}:layout={layout}[v]"
    out_mp4 = root.parent / f"{sid}_emotions.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", fc, "-map", "[v]", "-map", "0:a",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "22", "-c:a", "aac", "-shortest",
                    str(out_mp4)], check=True)
    # 정지 화면: 입이 열린 순간과 말이 끝난 뒤(감정만 남은 상태)를 위아래로
    out_png = root.parent / f"{sid}_emotions.png"
    stills = []
    for k, t in enumerate(args.still_times):
        p = root.parent / f"_{sid}_still{k}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(out_mp4), "-frames:v", "1", str(p)], check=True)
        stills.append(p)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *sum((["-i", str(p)] for p in stills), []),
                    "-filter_complex", "".join(f"[{i}:v]" for i in range(len(stills))) + f"vstack=inputs={len(stills)}",
                    str(out_png)], check=True)
    for p in stills:
        p.unlink()
    print(f"저장: {out_mp4.relative_to(ROOT)}, {out_png.relative_to(ROOT)} (정지 화면 {args.still_times}초)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
