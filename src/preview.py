"""시연 자료: 2D 입모양 미리보기 MP4 + 채널별 커브 그래프.

Maya나 ARKit 캐릭터 없이 결과를 눈으로 확인하기 위한 용도다. 입모양은 블렌드쉐입 값에서 직접 그린다.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .align import AlignResult
from .viseme import ARKIT_52, CH_INDEX

FONT_PATHS = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "C:/Windows/Fonts/malgun.ttf"]
W, H = 960, 540
BG = (30, 32, 38)
SKIN = (238, 200, 170)
LIP = (196, 92, 98)
INNER = (60, 20, 28)
TEETH = (245, 245, 240)
TONGUE = (220, 110, 120)
ACCENT = (120, 200, 255)
TEXT = (230, 230, 230)
MUTED = (140, 145, 155)

BAR_CHANNELS = [
    ("jawOpen", ["jawOpen"]),
    ("mouthFunnel", ["mouthFunnel"]),
    ("mouthPucker", ["mouthPucker"]),
    ("mouthSmile", ["mouthSmileLeft", "mouthSmileRight"]),
    ("mouthStretch", ["mouthStretchLeft", "mouthStretchRight"]),
    ("mouthPress", ["mouthPressLeft", "mouthPressRight"]),
    ("mouthLowerDown", ["mouthLowerDownLeft", "mouthLowerDownRight"]),
    ("mouthRollLower", ["mouthRollLower"]),
    ("tongueOut", ["tongueOut"]),
]


def _font(size: int) -> ImageFont.ImageFont:
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _v(frame: np.ndarray, *names: str) -> float:
    return float(np.mean([frame[CH_INDEX[n]] for n in names]))


def draw_mouth(d: ImageDraw.ImageDraw, frame: np.ndarray, cx: float, cy: float, scale: float = 1.0) -> None:
    jaw = _v(frame, "jawOpen")
    funnel = _v(frame, "mouthFunnel")
    pucker = _v(frame, "mouthPucker")
    smile = _v(frame, "mouthSmileLeft", "mouthSmileRight")
    stretch = _v(frame, "mouthStretchLeft", "mouthStretchRight")
    press = _v(frame, "mouthPressLeft", "mouthPressRight")
    lower = _v(frame, "mouthLowerDownLeft", "mouthLowerDownRight")
    roll = _v(frame, "mouthRollLower")
    tongue = _v(frame, "tongueOut")

    width = 150 * scale * (1 + 0.25 * smile + 0.3 * stretch - 0.5 * pucker - 0.25 * funnel)
    gap = 95 * scale * (jaw + 0.3 * lower + 0.12 * funnel)          # 입 안쪽이 벌어진 높이
    thick = 16 * scale * (1 + 0.7 * pucker + 0.4 * funnel - 0.45 * press)
    k = 0.5 if (funnel + pucker) > 0.3 else 0.85                      # 둥근 입 vs 옆으로 긴 입
    corner_up = 14 * scale * smile
    cy_low = cy + 0.15 * gap                                           # 턱이 내려가며 입 중심도 약간 내려감

    us = np.linspace(-1, 1, 41)
    base = np.clip(1 - us ** 2, 0, 1)
    xs = cx + us * width / 2
    g = gap / 2 * base ** k
    t = thick * base ** 0.5 + 2 * scale
    corner = -corner_up * us ** 2
    up_in = cy_low - g + corner
    up_out = up_in - t
    lo_in = cy_low + g + corner
    lo_out = lo_in + t * (1 - 0.6 * roll)

    if gap > 2:
        d.polygon(list(zip(xs, up_in)) + list(zip(xs[::-1], lo_in[::-1])), fill=INNER)
        teeth_h = min(gap * 0.25, 10 * scale)
        d.polygon(list(zip(xs * 0.9 + cx * 0.1, up_in)) + list(zip((xs * 0.9 + cx * 0.1)[::-1], (up_in + teeth_h)[::-1])), fill=TEETH)
    if tongue > 0.02:
        tw = width * 0.35
        d.ellipse([cx - tw / 2, cy_low + g.max() - 10 * scale, cx + tw / 2, cy_low + g.max() + 18 * scale * tongue + 6], fill=TONGUE)
    d.polygon(list(zip(xs, up_out)) + list(zip(xs[::-1], up_in[::-1])), fill=LIP)
    d.polygon(list(zip(xs, lo_in)) + list(zip(xs[::-1], lo_out[::-1])), fill=LIP)
    if gap <= 2:
        d.line(list(zip(xs, (up_in + lo_in) / 2)), fill=INNER, width=max(1, int(2 * scale)))


def _active_word(res: AlignResult, t: float) -> str:
    for w in res.words:
        if w.start <= t <= w.end:
            return w.text
    return ""


def _active_key(keys: list[dict], t: float) -> dict | None:
    """화면에 보이는 입모양에 가장 가까운(시간상 가장 가까운) 키."""
    if not keys:
        return None
    return min(keys, key=lambda k: abs(k["time"] - t))


def _timeline_background(frames: np.ndarray, res: AlignResult, fps: int, title: str) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # 얼굴
    fx, fy = 250, 235
    d.ellipse([fx - 170, fy - 200, fx + 170, fy + 200], fill=SKIN)
    for ex in (fx - 65, fx + 65):
        d.ellipse([ex - 22, fy - 70, ex + 22, fy - 50], fill=(255, 255, 255))
        d.ellipse([ex - 9, fy - 69, ex + 9, fy - 51], fill=(50, 40, 40))
    d.line([(fx, fy - 30), (fx - 12, fy + 25), (fx + 6, fy + 28)], fill=(200, 150, 125), width=3)
    d.text((24, 16), title, font=_font(22), fill=TEXT)
    # 타임라인 (jawOpen 곡선 + 단어 구간)
    x0, x1, y0, y1 = 40, W - 40, H - 80, H - 20
    d.rectangle([x0, y0, x1, y1], outline=(70, 74, 84))
    n = frames.shape[0]
    xs = np.linspace(x0, x1, n)
    jaw = frames[:, CH_INDEX["jawOpen"]]
    press = frames[:, CH_INDEX["mouthPressLeft"]]
    d.line(list(zip(xs, y1 - jaw * (y1 - y0))), fill=ACCENT, width=2)
    d.line(list(zip(xs, y1 - press * (y1 - y0))), fill=(255, 140, 120), width=1)
    dur = max(n - 1, 1) / fps
    f12 = _font(12)
    for w in res.words:
        wx0 = x0 + (x1 - x0) * w.start / dur
        wx1 = x0 + (x1 - x0) * w.end / dur
        d.line([(wx0, y0), (wx0, y0 + 6)], fill=MUTED)
        d.text((wx0 + 2, y0 - 16), w.text, font=f12, fill=MUTED)
        d.line([(wx1, y1 - 4), (wx1, y1)], fill=MUTED)
    d.text((x0, y1 + 2), "jawOpen", font=f12, fill=ACCENT)
    d.text((x0 + 70, y1 + 2), "mouthPress", font=f12, fill=(255, 140, 120))
    return img


def render_video(path: Path, frames: np.ndarray, res: AlignResult, wav: Path, fps: int, title: str = "",
                 keys: list[dict] | None = None) -> None:
    bg = _timeline_background(frames, res, fps, title)
    n = frames.shape[0]
    f28, f16, f13 = _font(28), _font(16), _font(13)
    x0, x1, y0, y1 = 40, W - 40, H - 80, H - 20
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k", "-shortest", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n):
            t = i / fps
            img = bg.copy()
            d = ImageDraw.Draw(img)
            fr = frames[i]
            draw_mouth(d, fr, 250, 330)
            # 오른쪽 정보 패널
            px = 520
            d.text((px, 60), f"{t:5.2f}s   frame {i}", font=f16, fill=MUTED)
            d.text((px, 90), _active_word(res, t) or "·", font=f28, fill=TEXT)
            k = _active_key(keys, t) if keys else None
            if k:
                d.text((px, 132), f"viseme  {k['group']}  {k['symbol']}", font=f16, fill=ACCENT)
            for j, (label, chans) in enumerate(BAR_CHANNELS):
                by = 175 + j * 28
                v = _v(fr, *chans)
                d.text((px, by), label, font=f13, fill=MUTED)
                d.rectangle([px + 130, by + 2, px + 380, by + 16], outline=(70, 74, 84))
                d.rectangle([px + 130, by + 2, px + 130 + 250 * v, by + 16], fill=ACCENT)
                d.text((px + 388, by), f"{v:.2f}", font=f13, fill=TEXT)
            hx = x0 + (x1 - x0) * i / max(n - 1, 1)
            d.line([(hx, y0), (hx, y1)], fill=(255, 220, 90), width=2)
            proc.stdin.write(img.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode:
        raise RuntimeError(f"ffmpeg 실패 ({proc.returncode}): {path}")


def render_curves(path: Path, frames: np.ndarray, rms: np.ndarray, res: AlignResult, fps: int, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for p in FONT_PATHS:
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    t = np.arange(frames.shape[0]) / fps
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 2.4]})
    a1.fill_between(t, rms / max(rms.max(), 1e-6), color="#9aa4b1", alpha=0.6, label="RMS (정규화)")
    for w in res.words:
        a1.axvspan(w.start, w.end, color="#4c9be8", alpha=0.12)
        a1.text((w.start + w.end) / 2, 1.02, w.text, ha="center", va="bottom", fontsize=9)
    a1.set_ylim(0, 1.25)
    a1.set_ylabel("음량")
    a1.set_title(f"{title}  [{res.language}]  {res.transcript}", fontsize=11)

    shown = set()
    for i, name in enumerate(ARKIT_52):
        base = name[:-4] if name.endswith("Left") else name[:-5] if name.endswith("Right") else name
        if base in shown or frames[:, i].max() < 1e-3:
            continue
        shown.add(base)
        label = f"{base} (L/R)" if base != name else name
        lw = 2.2 if name == "jawOpen" else 1.3
        a2.plot(t, frames[:, i], label=label, linewidth=lw)
    for w in res.words:
        a2.axvspan(w.start, w.end, color="#4c9be8", alpha=0.06)
    a2.set_ylim(-0.02, 1.0)
    a2.set_xlabel("시간 (초)")
    a2.set_ylabel("블렌드쉐입 값")
    a2.legend(loc="upper right", fontsize=8, ncol=3)
    a2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
