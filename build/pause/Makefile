.PHONY:	build push

IMAGE = pause
TAG = 0.8.0

build:
	./prepare.sh
	docker build -t beta.gcr.io/google_containers/$(IMAGE):$(TAG) .

push:	build
	gcloud docker --server=beta.gcr.io push beta.gcr.io/google_containers/$(IMAGE):$(TAG)

all:	push
