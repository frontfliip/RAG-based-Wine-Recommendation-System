FROM python:3.11-slim

WORKDIR /opt/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

COPY ../data_processing/wine_data_final.csv /opt/app/data_processing/wine_data_final.csv
COPY app/ ./app
COPY streamlit_app.py ./

ENV PYTHONPATH="${PYTHONPATH}:/opt/app"

CMD ["python", "app/main.py"]