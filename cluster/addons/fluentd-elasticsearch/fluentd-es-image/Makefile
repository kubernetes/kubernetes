.PHONY:	build push

IMAGE = fluentd-elasticsearch
TAG = 1.12

build:	
	docker build -t beta.gcr.io/google_containers/$(IMAGE):$(TAG) .

push:	
	gcloud docker --server=beta.gcr.io push beta.gcr.io/google_containers/$(IMAGE):$(TAG)
