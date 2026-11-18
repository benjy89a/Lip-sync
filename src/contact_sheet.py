"""일괄 처리 결과를 한 장에: 대사마다 말하는 중간 장면 하나 + 감정 + 대사.

사용:
  python -m src.cli samples/audio --life --out outputs/audio_guided     # 대본 표(script.csv)의 감정대로 12줄
  python -m src.contact_sheet outputs/audio_guided
출력: <run>/contact_sheet.png
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .cli import ROOT
from .preview import FONT_PATHS

TW, TH = 375, 345  # 칸 하나 크기 (얼굴 500×460을 줄임)


def _font(size):
    p = next((p for p in FONT_PATHS if Path(p).exists()), None)
    return ImageFont.truetype(p, size) if p else ImageFont.load_default()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="일괄 처리 결과를 한 장에 모으기")
    ap.add_argument("run", type=Path, help="cli 출력 폴더 (미리보기 MP4가 있어야 함)")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args(argv)

    run = args.run.resolve()
    order = [r["id"] for r in csv.DictReader(open(ROOT / "samples" / "script.csv", encoding="utf-8"))]
    ids = [i for i in order if (run / i / f"{i}.json").exists()]
    ids += sorted(p.parent.name for p in run.glob("*/*.json") if p.parent.name not in ids)
    f_cap, f_small = _font(19), _font(15)

    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        for sid in ids:
            d = json.loads((run / sid / f"{sid}.json").read_text(encoding="utf-8"))
            m = d["meta"]
            words = d["words"]
            # 말하는 중간 무렵, 입이 가장 크게 열린 프레임 (표정과 입이 함께 보이도록)
            fps = m["fps"]
            jaw = d["curves"]["jawOpen"]
            if words:
                a = int(((words[0]["start"] + words[-1]["end"]) / 2 - 0.4) * fps)
                b = int(((words[0]["start"] + words[-1]["end"]) / 2 + 0.4) * fps)
                a, b = max(0, a), min(len(jaw) - 1, max(a + 1, b))
                fi = max(range(a, b + 1), key=lambda i: jaw[i])
            else:
                fi = len(jaw) // 2
            png = Path(tmp) / f"{sid}.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{fi / fps:.3f}", "-i", str(run / sid / f"{sid}_preview.mp4"),
                            "-frames:v", "1", "-vf", "crop=500:460:0:0", str(png)], check=True)
            im = Image.open(png).convert("RGB").resize((TW, TH))
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, TH - 62, TW, TH], fill=(0, 0, 0))
            emo = m.get("emotion", {})
            label = emo.get("label", "중립") if emo.get("parts") else "중립"
            src = {"script": "대본 표", "option": "옵션", "none": "감정 없음", "disabled": "끔"}.get(emo.get("source", ""), "")
            dr.text((10, TH - 58), f"{sid}   {label}", font=f_cap, fill=(255, 205, 120))
            dr.text((TW - 70, TH - 56), src, font=f_small, fill=(160, 165, 175))
            text = m.get("transcript", "")
            dr.text((10, TH - 30), text if len(text) <= 26 else text[:25] + "…", font=f_small, fill=(230, 230, 230))
            tiles.append(im)

    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TW, 44 + rows * TH), (30, 32, 38))
    ImageDraw.Draw(sheet).text((12, 10), f"{run.name}: 대본 표(script.csv)의 감정대로 일괄 생성한 {len(tiles)}개 대사 (말하는 중간 장면)",
                               font=_font(20), fill=(230, 230, 230))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * TW, 44 + (k // cols) * TH))
    out = run / "contact_sheet.png"
    sheet.save(out)
    print(f"저장: {out.relative_to(ROOT)} ({len(tiles)}개)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
