.PHONY:	build push

TAG = 2.0

ARCH = amd64
IMAGE = pause
# ARCH = arm
# IMAGE = pause-arm

build:
	./prepare.sh ${ARCH}
	docker build -t gcr.io/google_containers/$(IMAGE):$(TAG) .

push:	build
	gcloud docker --server=gcr.io push gcr.io/google_containers/$(IMAGE):$(TAG)

all:	push
