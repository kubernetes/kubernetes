.PHONY:	build push

IMAGE = fluentd-elasticsearch
TAG = 1.9

build:	
	docker build -t gcr.io/google_containers/$(IMAGE):$(TAG) .

push:	
	gcloud docker push gcr.io/google_containers/$(IMAGE):$(TAG)
