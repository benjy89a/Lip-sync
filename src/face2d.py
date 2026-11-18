"""2D 사람 얼굴 그리기: ARKit 52 값 한 프레임 → 얼굴 이미지.

미리보기와 과제 시연용이다. 실제 캐릭터는 아티스트가 만든 블렌드쉐입 모양을 따르므로,
여기의 반응 크기(픽셀)는 "값의 변화가 눈에 잘 보이도록" 정한 것이고 데이터 값과는 무관하다.

화면은 캐릭터를 정면에서 본 모습이라 화면 왼쪽이 캐릭터의 오른쪽(…Right 채널)이다.
선을 매끄럽게 하려고 SS배 크기로 그린 뒤 줄인다.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .viseme import CH_INDEX

SS = 2                      # 크게 그린 뒤 줄이는 배율
FW, FH = 500, 460           # 얼굴 이미지 크기 (원래 배율)
FX = 250                    # 얼굴 가운데 x
EY = 196                    # 눈 높이
MY = 338                    # 입 높이

BG = (30, 32, 38)
SKIN = (236, 198, 168)
SKIN_SHADE = (214, 172, 142)
SHADOW = (178, 128, 100)
HAIR = (52, 38, 30)
BROW = (70, 48, 34)
LASH = (38, 26, 22)
IRIS = (92, 66, 44)
LIP = (188, 92, 92)
LIP_DARK = (150, 64, 66)
INNER = (64, 22, 30)
TEETH = (246, 244, 236)
TONGUE = (214, 108, 116)


def _v(f: np.ndarray, *names: str) -> float:
    return float(np.mean([f[CH_INDEX[n]] for n in names]))


def _mix(a, b, t):
    t = float(np.clip(t, 0, 1))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _P(pts):
    """원래 배율 좌표 → 크게 그릴 좌표"""
    return [(float(x) * SS, float(y) * SS) for x, y in pts]


def _quad(p0, p1, p2, n=16):
    """2차 베지어 곡선 위의 점들"""
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.array, (p0, p1, p2))
    return list(map(tuple, (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2))


def _line(d, pts, color, width):
    d.line(_P(pts), fill=color, width=max(1, int(round(width * SS))), joint="curve")


# ---------- 머리 ----------

def _head(d, jaw: float):
    # 목과 귀 (머리 뒤)
    d.polygon(_P([(FX - 72, 380), (FX + 72, 380), (FX + 82, FH), (FX - 82, FH)]), fill=SKIN_SHADE)
    for s in (-1, 1):
        d.ellipse(_P([(FX + s * 163 - 16, 182), (FX + s * 163 + 16, 246)]), fill=SKIN_SHADE)
    # 얼굴 윤곽: 위는 둥글고 아래는 턱으로 좁아진다. 턱을 벌리면 턱선이 내려간다
    th = np.linspace(0, 2 * np.pi, 120)
    s = np.sin(th)
    ry = np.where(s < 0, 192, 196 + 30 * jaw)
    x = FX + 162 * np.cos(th) * (1 - 0.24 * np.clip(s, 0, None) ** 2)
    y = 232 + ry * s
    d.polygon(_P(zip(x, y)), fill=SKIN)
    # 머리카락
    top = [(FX + 166 * np.cos(a) * 1.02, 232 + 198 * np.sin(a)) for a in np.linspace(np.pi * 1.06, np.pi * 1.94, 30)]
    hairline = _quad((FX + 150, 150), (FX, 70), (FX - 150, 150), 20)
    d.polygon(_P(top + hairline), fill=HAIR)


# ---------- 눈썹·이마 ----------

def _brows(d, f):
    inner_up = _v(f, "browInnerUp")
    lift_all = 0.0
    for side, sign in (("Right", -1), ("Left", 1)):
        ex = FX + sign * 64
        outer_up = _v(f, f"browOuterUp{side}")
        down = _v(f, f"browDown{side}")
        lift_all = max(lift_all, 0.6 * inner_up + 0.6 * outer_up)
        base = EY - 40
        xi, yi = ex - sign * (30 - 15 * down), base - 48 * inner_up + 30 * down
        xo, yo = ex + sign * 36, base - 38 * outer_up + 8 * down
        xm, ym = ex + sign * 7, (yi + yo) / 2 - 8 - 5 * outer_up + 5 * down
        top = _quad((xi, yi - 6.5), (xm, ym - 6), (xo, yo - 2.5), 14)
        bot = _quad((xo, yo + 2.5), (xm, ym + 5), (xi, yi + 6.5), 14)
        d.polygon(_P(top + bot), fill=BROW)
    # 이마 가로 주름 (눈썹을 올리면), 미간 세로 주름 (눈썹을 내리면)
    if lift_all > 0.1:
        c = _mix(SKIN, SHADOW, lift_all * 1.2)
        for k, y in enumerate((112, 126, 140)):
            span = 70 - 8 * k
            _line(d, _quad((FX - span, y + 3), (FX, y - 4), (FX + span, y + 3)), c, 1 + 1.5 * lift_all)
    down = _v(f, "browDownLeft", "browDownRight")
    if down > 0.1:
        c = _mix(SKIN, SHADOW, down * 1.3)
        for dx in (-6, 6):
            _line(d, [(FX + dx, EY - 46), (FX + dx * 0.5, EY - 26)], c, 1 + 2 * down)


# ---------- 눈 ----------

def _eyes(img, d, f):
    for side, sign in (("Right", -1), ("Left", 1)):
        ex, ey = FX + sign * 64, EY
        wide = _v(f, f"eyeWide{side}")
        squint = _v(f, f"eyeSquint{side}")
        cheek = _v(f, f"cheekSquint{side}")
        blink = _v(f, f"eyeBlink{side}")
        look_in, look_out = _v(f, f"eyeLookIn{side}"), _v(f, f"eyeLookOut{side}")
        look_up, look_down = _v(f, f"eyeLookUp{side}"), _v(f, f"eyeLookDown{side}")

        upper_h = 17 * (1 + 1.1 * wide) * max(0.0, 1 - 0.75 * squint) * (1 - blink) + 2 * look_up - 4 * look_down
        lower_h = 10 * max(0.0, 1 - 0.6 * squint - 0.75 * cheek) * (1 - blink)
        hw = 35
        xs = np.linspace(-1, 1, 27)
        base = np.clip(1 - xs ** 2, 0, 1)
        px = ex + xs * hw
        close_y = ey + 2
        up = close_y - max(upper_h, 0) * base ** 0.55 - 2 * sign * xs * 0   # 위 눈꺼풀
        lo = close_y + lower_h * base ** 0.75                              # 아래 눈꺼풀
        upper = list(zip(px, up))
        lower = list(zip(px, lo))
        if upper_h + lower_h > 1.5:
            # 흰자 + 눈동자 (흰자 모양으로 잘라서 그림)
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).polygon(_P(upper + lower[::-1]), fill=255)
            layer = Image.new("RGB", img.size, (250, 250, 248))
            ld = ImageDraw.Draw(layer)
            # 시선: 캐릭터 왼쪽을 보면 화면 오른쪽으로. 왼눈은 Out, 오른눈은 In이 캐릭터 왼쪽 방향
            gx = (look_out - look_in) if side == "Left" else (look_in - look_out)
            ix, iy = ex + 20 * gx, ey + 1 + 12 * (look_down - look_up)
            r = 15
            ld.ellipse(_P([(ix - r, iy - r), (ix + r, iy + r)]), fill=IRIS)
            ld.ellipse(_P([(ix - 6.5, iy - 6.5), (ix + 6.5, iy + 6.5)]), fill=(20, 14, 12))
            ld.ellipse(_P([(ix + 2.5, iy - 7.5), (ix + 6.5, iy - 3.5)]), fill=(255, 255, 255))
            img.paste(layer, (0, 0), mask)
        # 쌍꺼풀 선 (눈을 크게 뜨면 위로)
        crease = [(x, y - 6 - 3 * wide + 2 * squint) for x, y in upper[3:-3]]
        if upper_h > 3:
            _line(d, crease, SHADOW, 1.2)
        # 속눈썹 선과 아래 눈꺼풀
        _line(d, upper, LASH, 3.2)
        _line(d, lower[4:-4], _mix(SKIN, SHADOW, 0.6), 1.2)
        # 볼이 올라가면 눈 밑 볼살 선
        if cheek > 0.05:
            yb = close_y + lower_h + 9
            _line(d, _quad((ex - 24, yb + 2), (ex, yb - 5 * cheek), (ex + 24, yb + 2)), _mix(SKIN, SHADOW, 0.3 + cheek), 1 + 2.5 * cheek)


# ---------- 코 ----------

def _nose(d, f):
    sneer = _v(f, "noseSneerLeft", "noseSneerRight")
    lift = 9 * sneer
    flare = 1 + 0.45 * sneer
    _line(d, _quad((FX - 13, EY + 6), (FX - 17, 238), (FX - 20, 258 - lift)), _mix(SKIN, SHADOW, 0.45), 1.6)
    # 콧방울과 콧구멍
    for s in (-1, 1):
        wx, wy = FX + s * 19 * flare, 270 - lift
        _line(d, _quad((wx - s * 2, wy - 12), (wx + s * 9, wy - 2), (wx - s * 3, wy + 7)), SHADOW, 2)
        nx, ny = FX + s * 9 * flare, 276 - lift
        d.ellipse(_P([(nx - 5 * flare, ny - 3), (nx + 5 * flare, ny + 3)]), fill=(120, 76, 62))
    _line(d, _quad((FX - 9, 270 - lift), (FX, 266 - lift), (FX + 9, 270 - lift)), _mix(SKIN, SHADOW, 0.5), 1.4)
    # 코를 찡그리면 콧등 주름
    if sneer > 0.08:
        c = _mix(SKIN, SHADOW, 0.4 + sneer)
        for k in range(3):
            y = EY + 22 + k * 8
            _line(d, _quad((FX - 11, y + 2), (FX, y - 2), (FX + 11, y + 2)), c, 1 + 1.5 * sneer)


# ---------- 입 ----------

def _mouth(d, f) -> dict:
    jaw = _v(f, "jawOpen")
    funnel = _v(f, "mouthFunnel")
    pucker = _v(f, "mouthPucker")
    press = _v(f, "mouthPressLeft", "mouthPressRight")
    lower = _v(f, "mouthLowerDownLeft", "mouthLowerDownRight")
    upper_up = _v(f, "mouthUpperUpLeft", "mouthUpperUpRight")
    roll = _v(f, "mouthRollLower")
    tongue = _v(f, "tongueOut")
    shrug = _v(f, "mouthShrugUpper", "mouthShrugLower")

    def side_vals(side):
        smile = _v(f, f"mouthSmile{side}")
        stretch = _v(f, f"mouthStretch{side}")
        frown = _v(f, f"mouthFrown{side}")
        dimple = _v(f, f"mouthDimple{side}")
        half_w = 54 * (1 + 0.32 * smile + 0.35 * stretch + 0.15 * dimple - 0.5 * pucker - 0.25 * funnel)
        corner_up = 26 * smile + 10 * dimple - 26 * frown
        return half_w, corner_up

    w_r, up_r = side_vals("Right")
    w_l, up_l = side_vals("Left")
    gap = 88 * (jaw + 0.3 * lower + 0.12 * funnel)
    thick = 13 * (1 + 0.7 * pucker + 0.4 * funnel - 0.5 * press)
    k = 0.5 if (funnel + pucker) > 0.3 else 0.85
    cy = MY + 0.18 * gap

    us = np.linspace(-1, 1, 49)
    base = np.clip(1 - us ** 2, 0, 1)
    xs = FX + np.where(us < 0, us * w_r, us * w_l)
    g = gap / 2 * base ** k
    corner = -np.where(us < 0, up_r, up_l) * us ** 2
    up_in = cy - g + corner - 5 * upper_up * base
    bow = 3.2 * np.exp(-(us / 0.13) ** 2) - 2.2 * np.exp(-((np.abs(us) - 0.28) / 0.12) ** 2)   # 윗입술 가운데 굴곡
    up_out = up_in - (thick * base ** 0.5 + 1.5) + bow * (1 - 0.6 * pucker)
    lo_in = cy + g + corner - 5 * shrug * base
    lo_out = lo_in + (thick * 1.25 * base ** 0.5 + 1.5) * (1 - 0.6 * roll)

    if gap > 2:
        d.polygon(_P(list(zip(xs, up_in)) + list(zip(xs[::-1], lo_in[::-1]))), fill=INNER)
        teeth_h = min(gap * 0.28, 11)
        tx = xs * 0.88 + FX * 0.12
        d.polygon(_P(list(zip(tx, up_in)) + list(zip(tx[::-1], (up_in + teeth_h)[::-1]))), fill=TEETH)
        if gap > 18:
            d.polygon(_P(list(zip(tx, lo_in - teeth_h * 0.5)) + list(zip(tx[::-1], lo_in[::-1]))), fill=TEETH)
    if tongue > 0.02:
        tw = (w_r + w_l) * 0.35
        d.ellipse(_P([(FX - tw / 2, cy + g.max() - 10), (FX + tw / 2, cy + g.max() + 18 * tongue + 6)]), fill=TONGUE)
    d.polygon(_P(list(zip(xs, up_out)) + list(zip(xs[::-1], up_in[::-1]))), fill=LIP_DARK)
    d.polygon(_P(list(zip(xs, lo_in)) + list(zip(xs[::-1], lo_out[::-1]))), fill=LIP)
    _line(d, list(zip(xs, up_in if gap > 2 else (up_in + lo_in) / 2)), INNER, 1.6 if gap <= 2 else 1)
    # 턱 주름 (아랫입술 아래). 턱을 올리면(AU17) 진해진다
    cl = lo_out.max() + 16 + 4 * jaw
    _line(d, _quad((FX - 24, cl + 2), (FX, cl - 4 - 4 * shrug), (FX + 24, cl + 2)), _mix(SKIN, SHADOW, 0.25 + shrug), 1.2 + shrug)
    return {"right": (xs[0], up_in[0]), "left": (xs[-1], up_in[-1])}


def _folds(d, f, corners: dict):
    """팔자 주름: 콧방울 옆에서 입꼬리 바깥으로. 웃거나 코를 찡그리면 진해지고 입꼬리를 따라 움직인다"""
    sneer = _v(f, "noseSneerLeft", "noseSneerRight")
    for side, sign, key in (("Right", -1, "right"), ("Left", 1, "left")):
        smile = _v(f, f"mouthSmile{side}")
        stretch = _v(f, f"mouthStretch{side}")
        upper_up = _v(f, f"mouthUpperUp{side}")
        vis = float(np.clip(0.2 + 1.1 * smile + 0.6 * sneer + 0.5 * upper_up + 0.3 * stretch, 0, 1))
        cx, cy = corners[key]
        start = (FX + sign * 30, 268 - 8 * sneer)
        end = (cx + sign * 9, cy + 10)
        ctrl = (FX + sign * (52 + 18 * smile), (start[1] + end[1]) / 2 - 4)
        _line(d, _quad(start, ctrl, end), _mix(SKIN, SHADOW, vis), 1 + 2.2 * vis)
    # 볼이 올라가면 볼 위쪽에 홍조
    return


def _cheeks(img, f):
    cheek = _v(f, "cheekSquintLeft", "cheekSquintRight")
    smile = _v(f, "mouthSmileLeft", "mouthSmileRight")
    a = float(np.clip(0.6 * cheek + 0.3 * smile, 0, 1))
    if a < 0.05:
        return
    over = Image.new("RGB", img.size, (232, 140, 130))
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    for s in (-1, 1):
        cx, cy = FX + s * 92, 262 - 10 * cheek
        md.ellipse(_P([(cx - 34, cy - 20), (cx + 34, cy + 20)]), fill=int(70 * a))
    img.paste(over, (0, 0), mask)


def draw_face(frame: np.ndarray) -> Image.Image:
    """ARKit 52 값 한 프레임 → 얼굴 이미지 (FW×FH)"""
    img = Image.new("RGB", (FW * SS, FH * SS), BG)
    d = ImageDraw.Draw(img)
    _head(d, _v(frame, "jawOpen"))
    _cheeks(img, frame)
    d = ImageDraw.Draw(img)
    _brows(d, frame)
    _eyes(img, d, frame)
    d = ImageDraw.Draw(img)
    _nose(d, frame)
    corners = _mouth(d, frame)
    _folds(d, frame, corners)
    return img.resize((FW, FH), Image.LANCZOS)
