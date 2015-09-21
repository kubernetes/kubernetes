.PHONY:	build push

IMAGE = pause
TAG = 0.8.0

build:
	./prepare.sh
	docker build -t gcr.io/google_containers/$(IMAGE):$(TAG) .

push:	build
	gcloud docker push gcr.io/google_containers/$(IMAGE):$(TAG)

all:	push
