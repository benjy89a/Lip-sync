"""단어 → 음소 단위(PhoneUnit) 변환.

- 한국어: 유니코드 계산으로 초성/중성/종성 분해. 이중모음은 설정 파일에 따라 두 단계로 나눈다.
  받침은 7종성 대표음으로 읽는다(음절 경계를 넘는 발음 규칙은 g2pk 추가 시도에서 비교).
- 영어: g2p_en으로 ARPAbet 음소 변환.
- 한국어 문장 안의 영어 단어(예: "crazy했어")는 라틴 문자 부분만 영어 규칙으로 처리한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
        "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


@dataclass
class PhoneUnit:
    symbol: str          # 자모 또는 ARPAbet 음소
    kind: str            # vowel | glide | consonant | final
    group: str | None    # 비짐 그룹 (None이면 키를 만들지 않고 앞뒤 값을 이어받음)
    weight: float        # 단어 구간 안에서 시간을 나눌 때의 가중치
    lang: str


def decompose_hangul(ch: str) -> tuple[str, str, str] | None:
    code = ord(ch) - 0xAC00
    if not 0 <= code < 11172:
        return None
    return CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]


def korean_units(text: str, cfg: dict) -> list[PhoneUnit]:
    w = cfg["weights"]
    units: list[PhoneUnit] = []
    for ch in text:
        parts = decompose_hangul(ch)
        if parts is None:
            continue
        cho, jung, jong = parts
        if cho != "ㅇ":  # 초성 ㅇ은 소리가 없다
            units.append(PhoneUnit(cho, "consonant", cfg["consonants"].get(cho), w["consonant"], "ko"))
        groups = cfg["vowels"][jung]
        for i, g in enumerate(groups):
            is_glide = len(groups) > 1 and i == 0
            units.append(PhoneUnit(jung, "glide" if is_glide else "vowel", g,
                                   w["glide"] if is_glide else w["vowel"], "ko"))
        if jong:
            sound = cfg.get("final_neutralization", {}).get(jong, jong)
            units.append(PhoneUnit(sound, "final", cfg["consonants"].get(sound), w["final"], "ko"))
    return units


@lru_cache(maxsize=1)
def _g2p():
    from g2p_en import G2p

    return G2p()


def english_units(text: str, cfg: dict) -> list[PhoneUnit]:
    w = cfg["weights"]
    units: list[PhoneUnit] = []
    for ph in _g2p()(text):
        ph = re.sub(r"\d", "", ph).strip()
        if not ph or not ph.isalpha():
            continue
        if ph in cfg["vowels"]:
            groups = cfg["vowels"][ph]
            for i, g in enumerate(groups):
                # 영어 이중모음(AY, OW ...)은 앞이 주 모음이고 뒤가 미끄러지는 소리
                is_glide = len(groups) > 1 and i == len(groups) - 1
                units.append(PhoneUnit(ph, "glide" if is_glide else "vowel", g,
                                       w["glide"] if is_glide else w["vowel"], "en"))
        else:
            units.append(PhoneUnit(ph, "consonant", cfg["consonants"].get(ph), w["consonant"], "en"))
    return units


_CHUNK = re.compile(r"[가-힣]+|[A-Za-z']+")


def word_units(word: str, lang: str, cfgs: dict[str, dict]) -> list[PhoneUnit]:
    """단어 하나를 음소 단위 목록으로. 문자 종류별로 잘라서 한글은 한국어, 라틴 문자는 영어 규칙을 쓴다."""
    units: list[PhoneUnit] = []
    for chunk in _CHUNK.findall(word):
        if "가" <= chunk[0] <= "힣":
            units += korean_units(chunk, cfgs["ko"])
        else:
            units += english_units(chunk, cfgs["en"])
    return units
