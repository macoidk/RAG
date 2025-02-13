FROM python:3.12

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY code/ /app/code/
COPY data.py /app/
COPY model/ /app/model/
COPY data/ /app/data/
COPY embeddings/ /app/embeddings/
COPY tokenizer/ /app/tokenizer/
COPY .env /app/

RUN mkdir -p db
RUN mkdir -p dataset

COPY backend/ /app/backend/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]