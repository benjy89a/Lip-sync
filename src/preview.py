"""시연 자료: 2D 얼굴 미리보기 MP4 + 채널별 커브 그래프.

Maya나 ARKit 캐릭터 없이 결과를 눈으로 확인하기 위한 용도다. 얼굴은 블렌드쉐입 값에서 직접 그린다 (src/face2d.py).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .align import AlignResult
from .face2d import draw_face
from .viseme import ARKIT_52, CH_INDEX

FONT_PATHS = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "C:/Windows/Fonts/malgun.ttf"]
W, H = 960, 540
BG = (30, 32, 38)
ACCENT = (120, 200, 255)
EMO = (255, 190, 110)
LIFE = (150, 220, 160)
TEXT = (230, 230, 230)
MUTED = (140, 145, 155)

BAR_CHANNELS = [
    ("jawOpen", ["jawOpen"]),
    ("mouthPucker", ["mouthPucker"]),
    ("mouthSmile", ["mouthSmileLeft", "mouthSmileRight"]),
    ("mouthFrown", ["mouthFrownLeft", "mouthFrownRight"]),
    ("mouthPress", ["mouthPressLeft", "mouthPressRight"]),
    ("browInnerUp", ["browInnerUp"]),
    ("browOuterUp", ["browOuterUpLeft", "browOuterUpRight"]),
    ("browDown", ["browDownLeft", "browDownRight"]),
    ("eyeWide", ["eyeWideLeft", "eyeWideRight"]),
    ("eyeSquint", ["eyeSquintLeft", "eyeSquintRight"]),
    ("cheekSquint", ["cheekSquintLeft", "cheekSquintRight"]),
    ("noseSneer", ["noseSneerLeft", "noseSneerRight"]),
    ("eyeBlink", ["eyeBlinkLeft", "eyeBlinkRight"]),
    ("eyeLook (시선)", ["eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft"]),
]
UPPER_PREFIX = ("brow", "eye", "cheek", "nose")
LIFE_PREFIX = ("eyeBlink", "eyeLook")


def _font(size: int) -> ImageFont.ImageFont:
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _v(frame: np.ndarray, *names: str) -> float:
    return float(np.mean([frame[CH_INDEX[n]] for n in names]))


def _bar_value(frame: np.ndarray, label: str, chans: list[str]) -> float:
    if label.startswith("eyeLook"):  # 시선은 방향별 값 중 가장 큰 것 (얼마나 벗어났나)
        return float(max(frame[CH_INDEX[c]] for c in chans))
    return _v(frame, *chans)


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


def _timeline_background(frames: np.ndarray, res: AlignResult, fps: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # 타임라인 (jawOpen, mouthPress, 얼굴 위쪽 감정 + 단어 구간)
    x0, x1, y0, y1 = 40, W - 40, H - 80, H - 20
    d.rectangle([x0, y0, x1, y1], outline=(70, 74, 84))
    n = frames.shape[0]
    xs = np.linspace(x0, x1, n)
    jaw = frames[:, CH_INDEX["jawOpen"]]
    press = frames[:, CH_INDEX["mouthPressLeft"]]
    emo_idx = [i for i, c in enumerate(ARKIT_52) if c.startswith(UPPER_PREFIX) and not c.startswith(LIFE_PREFIX)]
    upper = frames[:, emo_idx].max(axis=1)
    blink = frames[:, CH_INDEX["eyeBlinkLeft"]]
    if upper.max() > 0:
        d.line(list(zip(xs, y1 - upper * (y1 - y0))), fill=EMO, width=1)
    if blink.max() > 0:
        d.line(list(zip(xs, y1 - blink * (y1 - y0))), fill=LIFE, width=1)
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
    if upper.max() > 0:
        d.text((x0 + 160, y1 + 2), "얼굴 위쪽(감정)", font=f12, fill=EMO)
    if blink.max() > 0:
        d.text((x0 + 270, y1 + 2), "eyeBlink", font=f12, fill=LIFE)
    return img


def render_video(path: Path, frames: np.ndarray, res: AlignResult, wav: Path, fps: int, title: str = "",
                 keys: list[dict] | None = None, emotion_label: str = "") -> None:
    bg = _timeline_background(frames, res, fps)
    n = frames.shape[0]
    f28, f22, f16, f13 = _font(28), _font(22), _font(16), _font(13)
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
            fr = frames[i]
            img.paste(draw_face(fr), (0, 0))
            d = ImageDraw.Draw(img)
            d.text((24, 16), title, font=f22, fill=TEXT)
            # 오른쪽 정보 패널
            px = 520
            d.text((px, 24), f"{t:5.2f}s   frame {i}", font=f16, fill=MUTED)
            d.text((px, 46), _active_word(res, t) or "·", font=f28, fill=TEXT)
            k = _active_key(keys, t) if keys else None
            if k:
                d.text((px, 86), f"viseme  {k['group']}  {k['symbol']}", font=f16, fill=ACCENT)
            if emotion_label:
                d.text((px + 200, 86), f"감정  {emotion_label}", font=f16, fill=EMO)
            for j, (label, chans) in enumerate(BAR_CHANNELS):
                by = 116 + j * 23
                v = _bar_value(fr, label, chans)
                color = LIFE if label.startswith(LIFE_PREFIX) else EMO if label.startswith(UPPER_PREFIX) else ACCENT
                d.text((px, by), label, font=f13, fill=MUTED)
                d.rectangle([px + 130, by + 2, px + 380, by + 15], outline=(70, 74, 84))
                d.rectangle([px + 130, by + 2, px + 130 + 250 * v, by + 15], fill=color)
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
    has = lambda pick: any(frames[:, i].max() > 1e-3 for i, c in enumerate(ARKIT_52) if pick(c))
    is_emo = lambda c: c.startswith(UPPER_PREFIX) and not c.startswith(LIFE_PREFIX)
    is_life = lambda c: c.startswith(LIFE_PREFIX)
    groups = [("입·턱", lambda c: not c.startswith(UPPER_PREFIX))]
    if has(is_emo):
        groups.append(("눈썹·눈·볼·코 (감정)", is_emo))
    if has(is_life):
        groups.append(("깜빡임·시선 (생기)", is_life))
    ratios = [1] + [2.2] + [1.6] * (len(groups) - 1)
    fig, axes = plt.subplots(len(groups) + 1, 1, figsize=(12, 3 + 2.3 * len(groups)), sharex=True,
                             gridspec_kw={"height_ratios": ratios})
    a1 = axes[0]
    a1.fill_between(t, rms / max(rms.max(), 1e-6), color="#9aa4b1", alpha=0.6)
    for w in res.words:
        a1.axvspan(w.start, w.end, color="#4c9be8", alpha=0.12)
        a1.text((w.start + w.end) / 2, 1.02, w.text, ha="center", va="bottom", fontsize=9)
    a1.set_ylim(0, 1.25)
    a1.set_ylabel("음량")
    a1.set_title(f"{title}  [{res.language}]  {res.transcript}", fontsize=11)

    def plot_group(ax, pick, ylabel):
        shown = set()
        for i, name in enumerate(ARKIT_52):
            if not pick(name) or frames[:, i].max() < 1e-3:
                continue
            base = name[:-4] if name.endswith("Left") else name[:-5] if name.endswith("Right") else name
            if base != name:
                pair = CH_INDEX.get(base + ("Right" if name.endswith("Left") else "Left"))
                same = pair is not None and np.allclose(frames[:, i], frames[:, pair])
                if same and base in shown:
                    continue
                label = f"{base} (L/R)" if same else name
                shown.add(base)
            else:
                label = name
            ax.plot(t, frames[:, i], label=label, linewidth=2.2 if name == "jawOpen" else 1.3)
        for w in res.words:
            ax.axvspan(w.start, w.end, color="#4c9be8", alpha=0.06)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8, ncol=3)

    for ax, (label, pick) in zip(axes[1:], groups):
        plot_group(ax, pick, label)
    axes[-1].set_xlabel("시간 (초)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
