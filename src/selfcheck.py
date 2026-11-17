"""채점 지표 자체 검증: 답을 아는 합성 라벨로 지표가 기대값을 내는지 확인한다.

실제 사람 라벨이 아니다. 파이프라인 예측값을 일부러 변형해 만든 라벨이라, 여기서 나온 수치는
"지표가 계산을 제대로 하는가"만 말해 주고 립싱크 품질과는 무관하다.

사용 (TTS 결과가 먼저 있어야 함):
  python -m src.cli samples/audio/tts
  python -m src.selfcheck
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .evaluate import Labels, human_agreement, mbp_metrics, onset_metrics, predicted_onsets, read_labels

ROOT = Path(__file__).resolve().parent.parent
FPS = 30


def main() -> int:
    src = ROOT / "outputs" / "tts" / "ko_01" / "ko_01.json"
    if not src.exists():
        print("먼저 실행: python -m src.cli samples/audio/tts")
        return 1
    d = json.loads(src.read_text(encoding="utf-8"))
    ons, names = predicted_onsets(d["phones"], with_names=True)
    mbp = [(p["start"], p["end"]) for p in d["phones"] if p["group"] == "MBP"]
    shift = 3 / FPS

    # 합성 라벨: 모음 시작을 3프레임 늦추고 4번째 모음은 빼고, 마지막 MBP는 엉뚱한 위치로 옮긴다
    lines = [f"{o + shift:.4f}\t{o + shift:.4f}\t{n}" for k, (o, n) in enumerate(zip(ons, names)) if k != 3]
    lines += [f"{s:.4f}\t{e:.4f}\tMBP" for s, e in mbp[:-1]] + [f"{d['meta']['duration'] - 0.3:.4f}\t{d['meta']['duration'] - 0.25:.4f}\tMBP"]
    tmp = Path(tempfile.mkdtemp())
    a = tmp / "ko_01.txt"
    a.write_text("\n".join(lines) + "\n", encoding="utf-8")
    labels: Labels = read_labels(a)

    om, _ = onset_metrics(labels, d["phones"], FPS)
    mm, _ = mbp_metrics(labels, d["curves"], FPS, tol_frames=1, jaw_max=0.1, press_min=0.3)
    checks = [
        ("모음 시작 평균 오차 = 3.0프레임", om["onset_mae_f"], 3.0),
        ("편향 = -3.0 (예측이 3프레임 빠름)", om["onset_bias_f"], -3.0),
        ("재현율 = 1.0 (라벨 13개 모두 짝지어짐)", om["onset_recall"], 1.0),
        (f"MBP 적중률 = {len(mbp) - 1}/{len(mbp)}", mm["mbp_hit_rate"], round((len(mbp) - 1) / len(mbp), 3)),
        ("잘못 닫힘 = 1 (라벨에서 뺀 마지막 MBP)", mm["mbp_false_closed"], 1),
    ]

    # 사람끼리 차이: 같은 라벨을 1프레임 민 사본과 비교하면 1.0프레임이 나와야 한다
    b = tmp / "ko_01_repeat.txt"
    b.write_text("\n".join(f"{float(s) + 1 / FPS:.4f}\t{float(e) + 1 / FPS:.4f}\t{n}"
                           for s, e, n in (l.split("\t") for l in lines)) + "\n", encoding="utf-8")
    checks.append(("사람끼리 차이 = 1.00프레임", "1.00프레임" in human_agreement(a, b, FPS), True))

    ok = True
    for name, got, want in checks:
        good = got == want
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}  (계산값 {got})")
    print("지표 자체 검증:", "모두 통과" if ok else "실패 있음")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
