FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/share/nltk_data \
    HF_HOME=/models/huggingface

# ffmpeg: 오디오 변환(m4a→wav)과 미리보기 MP4 합성
# fonts-nanum: 그래프/미리보기의 한글 표시
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-nanum libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements.lock.txt ./
# requirements.lock.txt: 따라 설치되는 패키지까지 전부 고정 (pip freeze 결과)
RUN pip install -r requirements.txt -c requirements.lock.txt

# g2p_en이 쓰는 NLTK 데이터를 빌드 시점에 받아 둔다 (실행 시 네트워크 불필요)
RUN python -c "import nltk; [nltk.download(p, download_dir='/usr/share/nltk_data') for p in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict']]"

CMD ["python", "-m", "src.cli", "--help"]
