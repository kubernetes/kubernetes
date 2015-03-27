.PHONY:	build push

IMAGE = fluentd-elasticsearch
TAG = 1.3

build:	
	docker build -t gcr.io/google-containers/$(IMAGE):$(TAG) .

push:	
	gcloud preview docker push gcr.io/google-containers/$(IMAGE):$(TAG)
