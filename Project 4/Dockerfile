FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (separate layer so it's cached on code-only changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

ENV PORT=8080

EXPOSE $PORT

CMD uvicorn server:app --host 0.0.0.0 --port $PORT
