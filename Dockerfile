FROM python:3.11-slim

WORKDIR /opt/app
COPY app/ ./


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

ENV PYTHONPATH="${PYTHONPATH}:/opt/app/app"

# Run the app
ENTRYPOINT ["python", "main.py"]