.PHONY:	build push

IMAGE = pause
TAG = 2.0

build:
	./prepare.sh
	docker build -t gcr.io/google_containers/$(IMAGE):$(TAG) .

push:	build
	gcloud docker --server=gcr.io push gcr.io/google_containers/$(IMAGE):$(TAG)

all:	push
