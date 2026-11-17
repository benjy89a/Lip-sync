"""효과 검증: 자동 결과를 수작업 라벨(정답)과 비교한다.

라벨 형식: Audacity 라벨 내보내기 (각 줄 `시작<TAB>끝<TAB>이름`)
  - 모음 시작: 점 라벨, 이름은 모음 (예: ㅏ, ㅓ, AA). 이름은 정답-예측 짝짓기에 쓴다 (입모양 그룹이 다르면 벌점)
  - 입 닫힘: 범위 라벨, 이름은 MBP (또는 ㅁ ㅂ ㅍ)
라벨 위치: samples/labels/<실행 이름>/<id>.txt 가 있으면 그것을, 없으면 samples/labels/<id>.txt
  (예: TTS 결과 outputs/tts 는 samples/labels/tts/ko_01.txt, 목소리 결과는 samples/labels/ko_01.txt)
수작업 시간: samples/labels/manual_time.csv (id,seconds[,note]) — 라벨 한 파일을 찍는 데 걸린 시간(초)

사용:
  python -m src.evaluate outputs/audio
  python -m src.evaluate outputs/tts outputs/audio      # 여러 실행 비교
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MBP_NAMES = {"mbp", "ㅁ", "ㅂ", "ㅍ", "ㅃ", "m", "b", "p"}
MAX_MBP_SEC = 0.6  # 이보다 긴 입 닫힘 라벨은 찍기 실수로 보고 제외


# ---------- 입력 읽기 ----------

@dataclass
class Labels:
    onsets: list[float]
    mbp: list[tuple[float, float]]
    onset_names: list[str] | None = None


def read_labels(path: Path) -> Labels:
    onsets, mbp = [], []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.startswith("\\"):  # '\'로 시작하는 줄은 Audacity 주파수 정보
            continue
        parts = line.split("\t")
        s, e = float(parts[0]), float(parts[1])
        name = parts[2].strip() if len(parts) > 2 else ""
        if name.lower() in MBP_NAMES:
            if e - s > MAX_MBP_SEC:
                # 입술 닫힘이 이렇게 길 수는 없다 (드래그 실수). 넣으면 대사 전체가 정답 구간이 되어 적중률이 무조건 올라간다.
                print(f"  [경고] {path.name}: {s:.2f}~{e:.2f}초 MBP 라벨({e - s:.2f}초)은 너무 길어 채점에서 제외")
                continue
            mbp.append((s, max(e, s)))
        else:
            onsets.append((s, name))
    onsets.sort()
    return Labels([t for t, _ in onsets], sorted(mbp), [n for _, n in onsets])


def find_labels(run_dir: Path, sid: str, labels_root: Path) -> Path | None:
    """라벨은 음성 출처 기준으로 찾는다. 실행 이름에 접미사가 붙어도(tts_rms, tts_small) 'tts' 라벨을 쓴다."""
    names = [run_dir.name, run_dir.name.split("_")[0]]
    dirs = [labels_root / n for n in dict.fromkeys(names)] + [labels_root]
    for p in (d / f"{sid}.txt" for d in dirs):
        if p.exists():
            return p
    return None


def read_csv_dict(path: Path, key: str = "id") -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as f:
        return {r[key]: r for r in csv.DictReader(f)}


# ---------- 지표 ----------

def normalize_text(s: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", s.lower())


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = normalize_text(ref), normalize_text(hyp)
    return levenshtein(r, h) / max(len(r), 1)


def predicted_onsets(phones: list[dict], with_names: bool = False, lead_sec: float = 0.0):
    """각 음절 모음 구간의 시작 시각 (단어 구간을 음소 가중치로 나눈 경계).

    주의: 커브에서 뽑은 값이 아니라 음소 구간 경계다. 키는 구간 중앙에 놓이므로 커브의 정점은 이보다 늦다.
    커브를 앞당기는 설정(lead_frames)은 lead_sec으로 반영한다.
    이중모음은 한 음절로 센다: 한국어는 glide→vowel 순서, 영어(ARPAbet)는 vowel→glide 순서.
    같은 모음이 이어지는 "하아", "아아"는 서로 다른 음절이므로 따로 센다.
    """
    out, names = [], []
    prev = None
    for p in phones:
        is_v = p["kind"] in ("vowel", "glide")
        same = prev is not None and prev["symbol"] == p["symbol"] and prev["word_index"] == p["word_index"]
        hangul = bool(p["symbol"]) and not p["symbol"].isascii()
        second_half = same and ((hangul and prev["kind"] == "glide" and p["kind"] == "vowel")
                                or (not hangul and prev["kind"] == "vowel" and p["kind"] == "glide"))
        if is_v and not second_half:
            out.append(p["start"] - lead_sec)
            names.append(p["symbol"])
        prev = p
    return (out, names) if with_names else out


_VOWEL_GROUPS: dict[str, str] | None = None


def _vowel_name(n: str) -> str | None:
    """모음 이름을 입모양 그룹으로 바꾼다. ㅐ와 ㅔ처럼 표기만 다르고 입모양이 같은 모음은 같은 것으로 본다."""
    global _VOWEL_GROUPS
    if _VOWEL_GROUPS is None:
        _VOWEL_GROUPS = {}
        for lang in ("ko", "en"):
            cfg = json.loads((ROOT / "config" / f"viseme_{lang}.json").read_text(encoding="utf-8"))
            _VOWEL_GROUPS.update({k: "+".join(v) for k, v in cfg["vowels"].items()})
    n = re.sub(r"\d", "", n.strip()).upper() if n.strip().isascii() else n.strip()
    return _VOWEL_GROUPS.get(n)


def match_monotonic(ref: list[float], hyp: list[float], gap_cost: float = 0.15,
                    ref_names: list[str] | None = None, hyp_names: list[str] | None = None,
                    mismatch_cost: float = 0.25) -> list[tuple[int, int]]:
    """순서를 지키며 정답-예측을 짝짓는다 (Needleman-Wunsch). 짝이 없으면 gap_cost(초)만큼 손해.

    한국어 모음은 0.1~0.2초 간격으로 촘촘해서 시간만 보면 한 칸씩 어긋나게 짝지어질 수 있다.
    라벨 이름이 모음(ㅏ, AA ...)이면 다른 모음끼리 짝지을 때 mismatch_cost를 더한다.
    """
    n, m = len(ref), len(hyp)
    rn = [_vowel_name(x) for x in ref_names] if ref_names else [None] * n
    hn = [_vowel_name(x) for x in hyp_names] if hyp_names else [None] * m

    def cost(i, j):
        c = abs(ref[i] - hyp[j])
        if rn[i] and hn[j] and rn[i] != hn[j]:
            c += mismatch_cost
        return c

    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1) * gap_cost
    D[0, :] = np.arange(m + 1) * gap_cost
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + cost(i - 1, j - 1), D[i - 1, j] + gap_cost, D[i, j - 1] + gap_cost)
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if np.isclose(D[i, j], D[i - 1, j - 1] + cost(i - 1, j - 1)):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif np.isclose(D[i, j], D[i - 1, j] + gap_cost):
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


MISS_CAP_SEC = 0.3  # 짝짓기가 포기하는 거리(gap_cost × 2). 놓친 모음은 이 오차로 계산한다.


def onset_metrics(labels: Labels, phones: list[dict], fps: int, lead_frames: int = 0) -> tuple[dict, list[dict]]:
    pred, pred_names = predicted_onsets(phones, with_names=True, lead_sec=lead_frames / fps)
    pairs = match_monotonic(labels.onsets, pred, ref_names=labels.onset_names, hyp_names=pred_names)
    errs = np.array([(pred[j] - labels.onsets[i]) * fps for i, j in pairs])  # +: 예측이 늦음
    detail = [{"label": round(labels.onsets[i], 3), "label_name": labels.onset_names[i], "pred": round(pred[j], 3),
               "pred_name": pred_names[j], "err_frames": round(float(e), 2)}
              for (i, j), e in zip(pairs, errs)]
    n = len(labels.onsets)
    if not errs.size:
        return {"onset_n": n, "onset_matched": 0, "onset_recall": 0.0,
                "onset_mae_capped_f": round(MISS_CAP_SEC * fps, 2) if n else None}, detail
    return {
        "onset_n": n,
        "onset_matched": len(pairs),
        # 평균 오차는 짝지어진 모음만으로 계산하므로, 모음을 적게 찾을수록 좋아 보인다.
        # 음량 기준선이 13개 중 6개만 찾고도 오차 1.81로 Whisper(3.0)보다 좋아 보인 사례 → 재현율을 함께 본다.
        "onset_recall": round(len(pairs) / n, 3),
        "onset_mae_f": round(float(np.abs(errs).mean()), 2),
        # 0.3초 넘게 어긋나면 짝이 안 지어져 위 평균에서 빠진다. 놓친 모음을 상한 오차(9프레임)로 넣은 평균을 함께 본다.
        "onset_mae_capped_f": round(float((np.abs(errs).sum() + (n - len(pairs)) * MISS_CAP_SEC * fps) / n), 2),
        "onset_median_f": round(float(np.median(np.abs(errs))), 2),
        "onset_bias_f": round(float(errs.mean()), 2),
        "onset_within2_rate": round(float((np.abs(errs) <= 2).sum() / n), 3),
    }, detail


def mbp_metrics(labels: Labels, curves: dict, fps: int, tol_frames: int, jaw_max: float, press_min: float) -> tuple[dict, list[dict]]:
    jaw = np.array(curves["jawOpen"])
    press = (np.array(curves["mouthPressLeft"]) + np.array(curves["mouthPressRight"])) / 2
    detail, hits = [], 0
    closed_all = (jaw <= jaw_max) & (press >= press_min)
    labeled = np.zeros(len(jaw), dtype=bool)
    for s, e in labels.mbp:
        a = max(0, int(np.floor(s * fps)) - tol_frames)
        b = min(len(jaw) - 1, int(np.ceil(e * fps)) + tol_frames)
        if a > b:  # 라벨이 음성 길이 밖
            detail.append({"start": s, "end": e, "hit": False, "note": "음성 길이 밖"})
            continue
        win = slice(a, b + 1)
        labeled[win] = True
        hit = bool(closed_all[win].any())
        hits += hit
        detail.append({"start": s, "end": e, "hit": hit, "min_jaw": round(float(jaw[win].min()), 3),
                       "max_press": round(float(press[win].max()), 3)})
    # 잘못 닫힘: 라벨된 구간과 전혀 겹치지 않는 닫힘 덩어리의 수 (놓친 것만 세면 "자주 닫을수록 유리"해진다)
    false_closed, i = 0, 0
    while i < len(jaw):
        if closed_all[i]:
            j = i
            while j + 1 < len(jaw) and closed_all[j + 1]:
                j += 1
            false_closed += not labeled[i:j + 1].any()
            i = j + 1
        else:
            i += 1
    n = len(labels.mbp)
    return {"mbp_n": n, "mbp_hit_rate": round(hits / n, 3) if n else None, "mbp_false_closed": int(false_closed)}, detail


# ---------- 실행 ----------

def evaluate_run(run_dir: Path, labels_root: Path, script: dict, manual: dict, args) -> list[dict]:
    summary = read_csv_dict(run_dir / "summary.csv")
    rows, details = [], {}
    for jpath in sorted(run_dir.glob("*/*.json")):
        sid = jpath.stem
        data = json.loads(jpath.read_text(encoding="utf-8"))
        fps = data["meta"]["fps"]
        row: dict = {"run": run_dir.name, "id": sid, "lang": data["meta"]["language"],
                     "duration": data["meta"]["duration"], "transcript": data["meta"]["transcript"]}
        # 전사가 없는 실행(음량 기준선)은 CER을 계산하지 않는다 — 1.0으로 넣으면 재려던 것(인식 오류)을 재지 않는 값이 됨
        if sid in script and data["meta"]["transcript"]:
            row["cer"] = round(cer(script[sid]["text"], data["meta"]["transcript"]), 3)
        s = summary.get(sid)
        if s:
            row["auto_sec"] = round(sum(float(s.get(k) or 0) for k in ("t_audio", "t_align", "t_viseme")), 2)
        if sid in manual:
            row["manual_sec"] = float(manual[sid]["seconds"])
            if "auto_sec" in row:
                row["time_saved_rate"] = round(1 - row["auto_sec"] / row["manual_sec"], 3)
        conf = data["meta"].get("confidence")
        if conf:
            row["min_word_prob"] = conf["min_word_prob"]
            row["needs_review"] = conf["needs_review"]
        # 라벨 없이 볼 수 있는 지표: 무음인데 입이 열려 있는 프레임 수
        wav = jpath.with_name(f"{sid}_16k.wav")
        if wav.exists():
            from .audio import load_wav
            from .boundary import silent_open_frames

            row["silent_open_f"] = silent_open_frames(load_wav(wav), data["curves"]["jawOpen"], fps, args.silence_threshold)
        lp = find_labels(run_dir, sid, labels_root)
        if lp:
            labels = read_labels(lp)
            om, od = onset_metrics(labels, data.get("phones", []), fps, data["meta"].get("lead_frames", 0))
            mm, md = mbp_metrics(labels, data["curves"], fps, args.mbp_tol_frames, args.jaw_max, args.press_min)
            row.update(om)
            row.update(mm)
            details[sid] = {"labels": str(lp), "onsets": od, "mbp": md}
        rows.append(row)
    out = ROOT / "outputs" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_dir.name}_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


COLS = ["run", "id", "lang", "duration", "cer", "onset_n", "onset_matched", "onset_recall", "onset_mae_f", "onset_mae_capped_f",
        "onset_median_f", "onset_bias_f", "onset_within2_rate", "mbp_n", "mbp_hit_rate", "mbp_false_closed", "silent_open_f", "manual_sec", "auto_sec", "time_saved_rate",
        "min_word_prob", "needs_review", "transcript"]


def to_markdown(rows: list[dict]) -> str:
    cols = [c for c in COLS if any(c in r for r in rows)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def aggregate(rows: list[dict]) -> list[dict]:
    """실행별 평균. 실패 사례(fail_)는 따로 보기 위해 정상 대사(ko_, en_)만 평균에 넣는다."""
    out = []
    for run in dict.fromkeys(r["run"] for r in rows):
        rs = [r for r in rows if r["run"] == run and not r["id"].startswith("fail_")]

        def mean(k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return round(float(np.mean(v)), 3) if v else None

        # 적중률/2프레임 비율은 파일별 평균이 아니라 전체 개수 기준으로 합친다
        mbp_n = sum(r.get("mbp_n", 0) for r in rs)
        mbp_hits = sum(round(r["mbp_hit_rate"] * r["mbp_n"]) for r in rs if r.get("mbp_hit_rate") is not None)
        out.append({"run": run, "id": f"평균({len(rs)}개, fail 제외)", "cer": mean("cer"),
                    "onset_recall": mean("onset_recall"), "onset_mae_f": mean("onset_mae_f"),
                    "onset_mae_capped_f": mean("onset_mae_capped_f"),
                    "silent_open_f": sum(r.get("silent_open_f") or 0 for r in rs),
                    "mbp_false_closed": sum(r.get("mbp_false_closed") or 0 for r in rs) if mbp_n else None, "onset_bias_f": mean("onset_bias_f"),
                    "onset_within2_rate": mean("onset_within2_rate"), "mbp_n": mbp_n or None,
                    "mbp_hit_rate": round(mbp_hits / mbp_n, 3) if mbp_n else None,
                    "manual_sec": mean("manual_sec"), "auto_sec": mean("auto_sec"),
                    "time_saved_rate": mean("time_saved_rate")})
    return out


def review_check(rows: list[dict], cer_err: float) -> str:
    """'검수 필요' 표시가 실제 인식 오류(CER >= cer_err)를 잡는지 교차표로 확인한다."""
    rs = [r for r in rows if "needs_review" in r and r.get("cer") is not None]
    if not rs:
        return ""
    tp = sum(r["needs_review"] and r["cer"] >= cer_err for r in rs)
    fp = sum(r["needs_review"] and r["cer"] < cer_err for r in rs)
    fn = [r for r in rs if not r["needs_review"] and r["cer"] >= cer_err]
    tn = len(rs) - tp - fp - len(fn)
    miss = ", ".join(f"{r['run']}/{r['id']}(CER {r['cer']})" for r in fn) or "없음"
    return (f"### 검수 표시 검증 (오류 = CER ≥ {cer_err}, {len(rs)}개)\n\n"
            f"| | 실제 오류 | 오류 아님 |\n|---|---|---|\n"
            f"| 검수 필요 표시 | {tp} | {fp} |\n| 표시 안 함 | {len(fn)} | {tn} |\n\n"
            f"놓친 오류: {miss}\n")


def human_agreement(a: Path, b: Path, fps: int = 30) -> str:
    """같은 파일을 두 번(또는 두 사람이) 라벨링한 결과의 차이 = 현행 수준(사람의 흔들림)."""
    la, lb = read_labels(a), read_labels(b)
    pairs = match_monotonic(la.onsets, lb.onsets, ref_names=la.onset_names, hyp_names=lb.onset_names)
    errs = np.array([(lb.onsets[j] - la.onsets[i]) * fps for i, j in pairs])
    mbp_pairs = match_monotonic([s for s, _ in la.mbp], [s for s, _ in lb.mbp])
    lines = [f"### 사람끼리의 차이: {a.name} vs {b}", "",
             f"- 모음 시작: {len(la.onsets)}개 vs {len(lb.onsets)}개, 짝지어짐 {len(pairs)}개"]
    if errs.size:
        lines += [f"- 평균 절대 차이 **{np.abs(errs).mean():.2f}프레임** (중앙값 {np.median(np.abs(errs)):.2f}, "
                  f"최대 {np.abs(errs).max():.2f}), 2프레임 이내 {(np.abs(errs) <= 2).mean():.0%}"]
    if mbp_pairs:
        d = np.array([abs(lb.mbp[j][0] - la.mbp[i][0]) * fps for i, j in mbp_pairs])
        lines += [f"- MBP 구간: {len(la.mbp)}개 vs {len(lb.mbp)}개, 시작 평균 차이 {d.mean():.2f}프레임"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="자동 립싱크 결과를 수작업 라벨과 비교")
    ap.add_argument("runs", type=Path, nargs="*", help="cli 출력 폴더 (예: outputs/audio outputs/tts)")
    ap.add_argument("--human", type=Path, nargs=2, metavar=("LABEL_A", "LABEL_B"),
                    help="같은 파일의 라벨 두 개를 비교해 사람끼리의 차이(현행 수준)를 잰다")
    ap.add_argument("--labels", type=Path, default=ROOT / "samples" / "labels")
    ap.add_argument("--mbp-tol-frames", type=int, default=1)
    ap.add_argument("--jaw-max", type=float, default=0.1, help="이 값 이하면 입이 닫힌 것으로 봄")
    ap.add_argument("--press-min", type=float, default=0.3)
    ap.add_argument("--cer-err", type=float, default=0.1, help="검수 표시 검증에서 '오류'로 볼 CER")
    ap.add_argument("--silence-threshold", type=float, default=0.12,
                    help="무음 판정 기준 (큰 소리 대비 비율). silent_open_f 계산에 사용")
    args = ap.parse_args(argv)

    if args.human:
        print(human_agreement(*args.human))
        return 0
    if not args.runs:
        ap.error("실행 폴더 또는 --human 라벨 두 개가 필요합니다")
    script = read_csv_dict(ROOT / "samples" / "script.csv")
    manual = read_csv_dict(args.labels / "manual_time.csv")
    rows = [r for run in args.runs for r in evaluate_run(run.resolve(), args.labels, script, manual, args)]
    if not rows:
        print("평가할 결과(JSON)가 없습니다.")
        return 1
    agg = aggregate(rows)

    out = ROOT / "outputs" / "eval"
    tag = "_vs_".join(r.resolve().name for r in args.runs)
    with (out / f"{tag}_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows + agg)
    md = "### 파일별\n\n" + to_markdown(rows) + "\n\n### 요약\n\n" + to_markdown(agg) + "\n\n" + review_check(rows, args.cer_err)
    (out / f"{tag}_metrics.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"저장: outputs/eval/{tag}_metrics.csv, .md, <run>_details.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
