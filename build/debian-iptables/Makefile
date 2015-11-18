.PHONY:	build push

IMAGE = debian-iptables
TAG = v1

build:
	docker build -t gcr.io/google_containers/$(IMAGE):$(TAG) .

push:	build
	gcloud docker --server=gcr.io push gcr.io/google_containers/$(IMAGE):$(TAG)

all:	push
