import os

from celery import Celery

# Get Kubernetes-provided address of the broker service
broker_service_host = os.environ.get('RABBITMQ_SERVICE_SERVICE_HOST')

app = Celery('tasks', broker='amqp://guest@%s//' % broker_service_host, backend='amqp')

@app.task
def add(x, y):
    return x + y

