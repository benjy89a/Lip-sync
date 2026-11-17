# 녹음·라벨링 가이드

수작업 라벨은 두 가지 역할을 한다.
1. **정답 데이터**: 자동 결과의 타이밍 정확도와 입 닫힘 적중률을 채점한다.
2. **수작업 기준 시간**: 라벨링에 걸린 시간을 "기존 방식"의 (보수적인) 작업 시간으로 쓴다.
   실제 키잉은 라벨링보다 오래 걸리므로, 이 기준으로도 단축되면 실제 단축 폭은 더 크다.

## 1. 녹음

| 항목 | 내용 |
|---|---|
| 대사 | `samples/script.csv`의 12줄 |
| 파일 이름 | ID 그대로: `ko_01.wav`, `en_02.m4a` … |
| 위치 | `samples/audio/` (TTS 임시 음성은 `samples/audio/tts/`에 따로 있음) |
| 형식 | wav 권장, 휴대폰 m4a도 가능 (자동 변환) |
| 환경 | 조용한 곳, 대사 앞뒤로 약 0.5초 여유 |

실패 사례는 연기가 중요하다.

| ID | 연기 |
|---|---|
| fail_01 | ko_01 대사를 **최대한 빠르게** |
| fail_02 | 한영 혼용: 영어 단어는 **영어 발음**으로 |
| fail_03 | 실제로 **웃고 한숨** 쉬듯이 |
| fail_04 | ko_03 대사를 **속삭이듯이** |

## 2. 라벨링 (Audacity)

권장 대상: **ko_01~05, en_01** (나머지는 정성 평가). 시간 여유가 있으면 en_02, en_03도.

> **권장: 전용 라벨러 `tools/labeler.html`** (Chrome에서 열기). Audacity로는 짧은 음성에서 찍기가 어려워 만들었다.
> 찍을 순서가 미리 나열돼 있어 클릭(모음)과 드래그(입 닫힘)만 하면 되고, 0.5×/0.25× 느린 재생, 확대, 스펙트로그램,
> 작업 시간 자동 측정, `samples/labels/`(다시 하기는 `samples/labels/repeat/`)에 바로 저장을 지원한다.
> 수작업 시간은 이 도구 기준이라 Audacity 현업 작업보다 짧게 나온다 → 자동화에 불리한, 보수적인 비교.
>
> **Audacity 버전 주의**: 이 PC에는 Audacity 4.0.2가 설치돼 있다. 아래 메뉴와 단축키는 3.x 기준이고,
> 4.0의 라벨 기능 공식 설명서는 아직 없다 (2026-11-17 확인). 4.0에서 위치가 다르면 메뉴에서 "Label"을 찾는다.
> **ko_01 하나를 먼저 끝내고 내보낸 파일이 채점에 읽히는지 확인한 뒤** 나머지를 진행한다.

### 대사별 찍을 것 (대본 기준)

괄호 안이 그 단어에서 찍을 모음 순서, `[MBP]`는 입 닫힘 구간이다. 실제 발음이 다르면 **들리는 대로** 찍는다.

| ID | 모음 수 | 입 닫힘 | 순서 |
|---|---|---|---|
| ko_01 | 14 | 6 | 바람이(`[MBP]` ㅏ ㅏ `[MBP]` ㅣ) / 멈추면(`[MBP]` ㅓ `[MBP]` ㅜ `[MBP]` ㅕ) / 우리는(ㅜ ㅣ ㅡ) / 바다로(`[MBP]` ㅏ ㅏ ㅗ) / 간다(ㅏ ㅏ) |
| ko_02 | 15 | 2 | 우리(ㅜ ㅣ) / 모두(`[MBP]` ㅗ ㅜ) / 그(ㅡ) / 오두막으로(ㅗ ㅜ `[MBP]` ㅏ ㅡ ㅗ) / 돌아가야(ㅗ ㅏ ㅏ ㅑ) / 해(ㅐ) |
| ko_03 | 10 | 0 | 이(ㅣ) / 길의(ㅣ ㅢ) / 끝에서(ㅡ ㅔ ㅓ) / 기다릴게(ㅣ ㅏ ㅣ ㅔ) |
| ko_04 | 13 | 1 | 네가(ㅔ ㅏ) / 말했던(`[MBP]` ㅏ ㅐ ㅓ) / 그(ㅡ) / 약속(ㅑ ㅗ) / 아직(ㅏ ㅣ) / 기억해(ㅣ ㅓ ㅐ) |
| ko_05 | 13 | 3~4 | 같이(ㅏ ㅣ) / 국물(ㅜ `[MBP]` ㅜ) / 좀(ㅗ `[MBP]`) / 먹자(`[MBP]` ㅓ ㅏ) / 좋아(ㅗ ㅏ) / 정말(ㅓ `[MBP]` ㅏ) / 좋아(ㅗ ㅏ) |
| en_01 | 9 | 3 | The(AH) / birch(`[MBP]` ER) / canoe(AH UW) / slid(IH) / on(AA) / the(AH) / smooth(`[MBP]` UW) / planks(`[MBP]` AE) |

- **붙어 있는 닫힘은 한 구간으로** 찍는다. 예: "좀 먹자"는 ㅁ 받침과 ㅁ 초성이 이어지므로 입이 한 번만 닫힌다 (ko_05가 3~4인 이유).
- "길의"처럼 실제로 [기레]에 가깝게 발음했다면 들리는 모음(ㅔ)으로 적는다.
- 음성 파일은 **원본**(`samples/audio/ko_01.wav`)을 연다. `outputs/` 안의 `_16k.wav`도 시간은 같다.

1. **스톱워치를 켠다.** 파일을 여는 순간부터 라벨 내보내기까지의 시간을 잰다.
2. Audacity에서 음성 파일을 연다. 파형 + 스펙트로그램 보기를 쓰면 편하다.
3. **모음 시작**: 각 음절의 모음이 시작되는 순간에 커서를 두고 `Ctrl+B` → 이름에 **모음 글자**를 적는다.
   - 한국어: `ㅏ` `ㅓ` `ㅗ` … 이중모음은 겉모양 그대로 (`ㅕ`, `ㅘ`). 받침은 찍지 않는다.
   - 영어: ARPAbet(`AA`, `IY` …)을 알면 그것을, 모르면 `V`라고만 적어도 된다 (시간만 비교).
   - 모음 이름은 채점 때 정답-예측 짝짓기에 쓰인다. 한국어 모음은 0.1~0.2초 간격으로 촘촘해서,
     이름이 없으면 한 칸씩 어긋나게 짝지어질 수 있다.
4. **입 닫힘 (ㅁ ㅂ ㅍ / M B P)**: 입술이 닫혀 있는 구간을 드래그로 선택 → `Ctrl+B` → 이름 `MBP`.
   - 받침 ㅁ(예: "멈추면"의 첫 ㅁ 받침)도 포함한다.
5. 파일 → 내보내기 → **라벨 내보내기** → `samples/labels/ko_01.txt`
6. 걸린 시간을 `samples/labels/manual_time.csv`에 적는다.

```csv
id,seconds,note
ko_01,420,
ko_02,380,
```

7. **현행 수준 측정 (사람끼리의 차이)**: 다른 파일을 모두 끝낸 뒤, **ko_01을 처음부터 한 번 더** 라벨링한다.
   첫 번째 라벨을 보지 않고 새로 찍어서 `samples/labels/repeat/ko_01.txt`로 저장한다.
   두 라벨의 차이가 "사람이 해도 이 정도는 흔들린다"는 현행 수준이고, 성공 기준(2프레임)의 근거가 된다.
   가능하면 다른 사람에게도 같은 파일을 부탁해 사람 간 차이도 잰다.

라벨 파일 예시 (Audacity가 만드는 형식, 탭 구분):

```
0.153000	0.153000	ㅏ
0.402000	0.471000	MBP
0.480000	0.480000	ㅏ
```

## 3. 실행 순서

```bash
# 목소리 → outputs/audio/
docker compose run --rm lipsync python -m src.cli samples/audio
# TTS → outputs/tts/
docker compose run --rm lipsync python -m src.cli samples/audio/tts
# 기준선: 음량만으로 턱 움직이기 → outputs/audio_rms/
docker compose run --rm lipsync python -m src.baseline samples/audio
# 채점 (TTS 라벨은 samples/labels/tts/<id>.txt 에 두면 함께 채점)
docker compose run --rm lipsync python -m src.evaluate outputs/audio outputs/audio_rms outputs/tts
# 현행 수준: 같은 파일 두 번 라벨링한 차이
docker compose run --rm lipsync python -m src.evaluate --human samples/labels/ko_01.txt samples/labels/repeat/ko_01.txt
# TTS vs 목소리 비교 영상/그래프 → outputs/compare/tts_vs_audio/
docker compose run --rm lipsync python -m src.compare outputs/tts outputs/audio --names TTS VOICE
```
