.PHONY:	build push

IMAGE = debian-iptables
TAG = v1

build:
	docker build -t beta.gcr.io/google_containers/$(IMAGE):$(TAG) .

push:	build
	gcloud docker --server=beta.gcr.io push beta.gcr.io/google_containers/$(IMAGE):$(TAG)

all:	push
