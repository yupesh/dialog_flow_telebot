# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster 

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ARG src="./ru_dataset_3/"
ARG target="/app/ru_dataset_3/"
COPY ${src} ${target}

ARG src="./pickle/"
ARG target="/app/pickle/"
COPY ${src} ${target}

# cache mfaq model
#RUN ["python3", "-c", "from sentence_transformers import SentenceTransformer; _ = SentenceTransformer('clips/mfaq')"]

COPY . .

CMD ["python3", "run.py"]
