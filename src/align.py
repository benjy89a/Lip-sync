"""정렬 모듈: 음성 → 단어 단위 타이밍.

인터페이스(`Aligner`)와 구현을 분리해, Whisper 대신 WhisperX나 Montreal Forced Aligner(MFA)로
교체할 수 있게 한다. 게임 대사는 대본이 이미 있으므로 실무에서는 대본을 이용하는 강제 정렬이 더 적합하다.
그래서 `align()`은 처음부터 `text`(대본)를 받을 수 있게 설계했다.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class AlignResult:
    language: str
    duration: float
    words: list[Word] = field(default_factory=list)
    transcript: str = ""
    aligner: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Aligner(ABC):
    """음성 파일에서 단어별 시작/끝 시각을 얻는 정렬기 인터페이스."""

    name: str = "base"

    @abstractmethod
    def align(self, wav_path: Path, language: str | None = None, text: str | None = None) -> AlignResult:
        """wav_path: 16kHz 모노 wav. language: 'ko'/'en'/None(자동 감지). text: 대본(선택)."""


class WhisperAligner(Aligner):
    """faster-whisper의 word_timestamps로 단어 타이밍을 얻는다."""

    def __init__(self, model_size: str = "small", device: str | None = None, compute_type: str | None = None):
        from faster_whisper import WhisperModel

        device = device or os.environ.get("LIPSYNC_DEVICE", "cpu")
        compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.model_size = model_size
        self.name = f"faster-whisper:{model_size}"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    SUPPORTED = ("ko", "en")

    def detect_language(self, wav_path: Path) -> tuple[str, dict[str, float]]:
        """지원 언어(ko/en) 중에서만 고른다.

        Whisper 자동 감지는 99개 언어 중에서 고르기 때문에, 억양이 단조로운 한국어를 일본어로 오인하는 경우가 있었다
        (TTS ko_02 → 'ja'로 감지되어 가타카나로 전사됨). 지원 언어의 확률만 비교하면 이 오류를 막을 수 있다.
        """
        from faster_whisper.audio import decode_audio

        _, _, all_probs = self.model.detect_language(decode_audio(str(wav_path)))
        probs = {lang: p for lang, p in all_probs if lang in self.SUPPORTED}
        best = max(self.SUPPORTED, key=lambda l: probs.get(l, 0.0))
        return best, {k: round(v, 4) for k, v in probs.items()}

    def align(self, wav_path: Path, language: str | None = None, text: str | None = None) -> AlignResult:
        if language is None:
            language, self.last_language_probs = self.detect_language(wav_path)
        segments, info = self.model.transcribe(
            str(wav_path),
            language=language,
            word_timestamps=True,
            beam_size=5,
            temperature=0.0,          # 재현성: 샘플링 없이 결정적으로 디코딩
            condition_on_previous_text=False,
            initial_prompt=text,      # 대본이 있으면 인식 힌트로 사용 (정렬 자체는 Whisper가 함)
        )
        words: list[Word] = []
        texts: list[str] = []
        for seg in segments:
            texts.append(seg.text.strip())
            for w in seg.words or []:
                words.append(Word(w.word.strip(), round(w.start, 3), round(w.end, 3), round(w.probability, 3)))
        return AlignResult(
            language=info.language,
            duration=round(info.duration, 3),
            words=words,
            transcript=" ".join(texts),
            aligner=self.name,
        )
