
TAG = v2.6.0-2
PREFIX = kubernetes

all: container

container:
	docker build -t $(PREFIX)/heapster_grafana:$(TAG) .

push:
	docker push $(PREFIX)/grafana:$(TAG)


