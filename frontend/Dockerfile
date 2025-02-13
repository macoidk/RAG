FROM python:3.12

WORKDIR /app

COPY frontend/requirementsl.txt .

RUN pip install --no-cache-dir -r requirementsl.txt

COPY frontend/ /app/frontend/
COPY .env /app/

EXPOSE 8001

CMD ["python", "frontend/manage.py", "runserver", "0.0.0.0:8001"]