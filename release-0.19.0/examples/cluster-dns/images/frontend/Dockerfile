FROM python:2.7

RUN pip install requests

COPY . /dns-frontend
WORKDIR /dns-frontend

CMD ["python", "client.py"]
