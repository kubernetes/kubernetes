FROM python:2.7-slim

RUN pip install requests

COPY . /dns-frontend
WORKDIR /dns-frontend

CMD ["python", "client.py"]
