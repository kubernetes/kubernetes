.PHONY:	build push

IMAGE = fluentd-elasticsearch
TAG = 1.12

build:	
	docker build -t gcr.io/google_containers/$(IMAGE):$(TAG) .

push:	
	gcloud docker --server=gcr.io push gcr.io/google_containers/$(IMAGE):$(TAG)
