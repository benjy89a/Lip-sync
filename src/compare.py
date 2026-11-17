"""두 실행 결과 비교 (예: TTS vs 실제 성우 목소리, 기준선 vs g2pk).

사용:
  python -m src.compare outputs/tts outputs/audio --names TTS VOICE            # 공통 ID 전부
  python -m src.compare outputs/tts outputs/audio --ids ko_01 ko_05
출력: outputs/compare/<A>_vs_<B>/<id>_compare.png, <id>_compare.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from .evaluate import cer, find_labels, read_csv_dict, read_labels
from .preview import FONT_PATHS

ROOT = Path(__file__).resolve().parent.parent


def _font_path() -> str | None:
    return next((p for p in FONT_PATHS if Path(p).exists()), None)


def compare_png(path: Path, sid: str, runs: list[Path], names: list[str], labels_root: Path, script: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fp = _font_path()
    if fp:
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(len(runs), 1, figsize=(12, 3.2 * len(runs)))
    for ax, run, name in zip(np.atleast_1d(axes), runs, names):
        d = json.loads((run / sid / f"{sid}.json").read_text(encoding="utf-8"))
        fps = d["meta"]["fps"]
        c = d["curves"]
        t = np.arange(len(c["jawOpen"])) / fps
        ax.plot(t, c["jawOpen"], label="jawOpen", lw=2, color="#2f7fd8")
        ax.plot(t, (np.array(c["mouthPressLeft"]) + np.array(c["mouthPressRight"])) / 2, label="mouthPress", color="#d86a4a")
        ax.plot(t, c["mouthPucker"], label="mouthPucker", color="#3aa66a", lw=1)
        for w in d["words"]:
            ax.axvspan(w["start"], w["end"], color="#4c9be8", alpha=0.08)
            ax.text((w["start"] + w["end"]) / 2, 0.97, w["text"], ha="center", va="top", fontsize=8)
        lp = find_labels(run, sid, labels_root)
        if lp:
            lab = read_labels(lp)
            for i, o in enumerate(lab.onsets):
                ax.axvline(o, color="#555", lw=0.8, ls=":", label="정답 모음 시작" if i == 0 else None)
            for i, (s, e) in enumerate(lab.mbp):
                ax.axvspan(s, e, color="#d86a4a", alpha=0.18, label="정답 MBP 구간" if i == 0 else None)
        extra = f"  CER {cer(script[sid]['text'], d['meta']['transcript']):.2f}" if sid in script else ""
        ax.set_title(f"[{name}] {d['meta']['transcript']}  ({d['meta']['duration']:.2f}s{extra})", fontsize=10, loc="left")
        ax.set_ylim(-0.02, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncol=5)
    np.atleast_1d(axes)[-1].set_xlabel("시간 (초)")
    fig.suptitle(sid, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def compare_mp4(path: Path, sid: str, runs: list[Path], names: list[str]) -> None:
    """A 미리보기 → B 미리보기를 이어 붙이고 각각 이름 배지를 단다."""
    inputs, filters, concat_in = [], [], []
    fp = _font_path()
    for i, (run, name) in enumerate(zip(runs, names)):
        inputs += ["-i", str(run / sid / f"{sid}_preview.mp4")]
        draw = (f"drawtext=fontfile={fp}:text='{name}':x=w-tw-24:y=18:fontsize=30:fontcolor=yellow:"
                f"box=1:boxcolor=black@0.5:boxborderw=8") if fp else "null"
        filters.append(f"[{i}:v]{draw}[v{i}]")
        concat_in.append(f"[v{i}][{i}:a]")
    fc = ";".join(filters) + ";" + "".join(concat_in) + f"concat=n={len(runs)}:v=1:a=1[v][a]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", "-c:a", "aac",
                    str(path)], check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="두 실행 결과를 나란히 비교")
    ap.add_argument("runs", type=Path, nargs=2)
    ap.add_argument("--names", nargs=2, default=None, help="표시 이름 (기본: 폴더 이름)")
    ap.add_argument("--ids", nargs="*", default=None, help="비교할 ID (기본: 두 실행의 공통 ID 전부)")
    ap.add_argument("--labels", type=Path, default=ROOT / "samples" / "labels")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)

    runs = [r.resolve() for r in args.runs]
    names = args.names or [r.name for r in runs]
    common = sorted(set(p.parent.name for p in runs[0].glob("*/*.json")) & set(p.parent.name for p in runs[1].glob("*/*.json")))
    ids = args.ids or common
    if not ids:
        print("두 실행에 공통으로 있는 ID가 없습니다.")
        return 1
    script = read_csv_dict(ROOT / "samples" / "script.csv")
    out = ROOT / "outputs" / "compare" / f"{runs[0].name}_vs_{runs[1].name}"
    out.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        compare_png(out / f"{sid}_compare.png", sid, runs, names, args.labels, script)
        if not args.no_video:
            compare_mp4(out / f"{sid}_compare.mp4", sid, runs, names)
        print(f"  {sid}: 완료")
    print(f"저장: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
